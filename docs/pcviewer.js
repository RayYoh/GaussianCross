/* Minimal WebGL point-cloud viewer for the GaussianCross page.
   No dependencies. Loads the packed .bin format (16-byte header):
     "GCPC" | uint32 count | uint32 nColorSets | uint32 flags
     | float32 xyz[3n] | uint8 rgb[nColorSets][3n]
     | int8 nrm[3n]      (iff flags & 1)
     | float32 query[3]  (iff flags & 2)   -- the similarity query point
   The trailing query is read through a DataView because the preceding uint8/int8
   blocks leave it at an arbitrary byte offset.
*/
(function (global) {
  'use strict';

  var VS = [
    'attribute vec3 aPos;',
    'attribute vec3 aCol;',
    'uniform mat4 uMVP;',
    'uniform float uSize;',
    'uniform float uMix;',        // 0 -> colour set A, 1 -> set B
    'uniform vec2 uGamma;',       // per-set gamma; <1 brightens
    'uniform vec3 uLight;',       // light direction (view space-ish)
    'uniform float uShade;',      // 0 = flat colour, 1 = full lambert
    'attribute vec3 aCol2;',
    'attribute vec3 aNrm;',
    'varying vec3 vCol;',
    'void main(){',
    '  vec4 p = uMVP * vec4(aPos,1.0);',
    '  gl_Position = p;',
    '  gl_PointSize = uSize / max(p.w, 0.01);',
    '  vec3 ca = pow(aCol,  vec3(uGamma.x));',
    '  vec3 cb = pow(aCol2, vec3(uGamma.y));',
    '  vec3 base = mix(ca, cb, uMix);',
    '  vec3 n = normalize(aNrm);',
    '  float d = abs(dot(n, normalize(uLight)));',      // two-sided: scans have flipped normals
    '  float lit = 0.55 + 0.75 * d;',
    '  lit = mix(1.0, lit, uShade * (1.0 - 0.55 * uMix));',  // shade RGB more than the feature field
    '  vCol = clamp(base * lit, 0.0, 1.0);',
    '}'
  ].join('\n');

  var FS = [
    'precision mediump float;',
    'varying vec3 vCol;',
    'void main(){',
    '  vec2 d = gl_PointCoord - vec2(0.5);',
    '  if (dot(d,d) > 0.25) discard;',        // round points
    '  gl_FragColor = vec4(vCol, 1.0);',
    '}'
  ].join('\n');

  /* The query marker is a screen-space crosshair: gl.lineWidth is clamped to 1 by
     every browser, so the bars are built from triangles offset in pixels around the
     projected query point. That also keeps it a constant size while you zoom. */
  var MVS = [
    'attribute vec3 aBar;',       // x = along (px), y = across (+-1), z = 0 horiz / 1 vert
    'uniform mat4 uMVP;',
    'uniform vec3 uQuery;',
    'uniform vec2 uPx;',          // clip units per pixel
    'uniform float uThick;',      // half-thickness in px
    'void main(){',
    '  vec4 p = uMVP * vec4(uQuery, 1.0);',
    '  vec2 o = (aBar.z < 0.5) ? vec2(aBar.x, aBar.y * uThick)',
    '                          : vec2(aBar.y * uThick, aBar.x);',
    '  p.xy += o * uPx * p.w;',
    '  gl_Position = p;',
    '}'
  ].join('\n');

  var MFS = [
    'precision mediump float;',
    'uniform vec4 uCol;',
    'void main(){ gl_FragColor = uCol; }'
  ].join('\n');

  // two bars x two triangles, in pixels: aBar = (along, across, isVertical)
  function crossGeom(len) {
    var v = [], L = len;
    [0, 1].forEach(function (vert) {
      v.push(-L,-1,vert,  L,-1,vert,  L,1,vert,
             -L,-1,vert,   L,1,vert, -L,1,vert);
    });
    return new Float32Array(v);
  }

  function compile(gl, type, src) {
    var s = gl.createShader(type);
    gl.shaderSource(s, src); gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
    return s;
  }

  // --- tiny mat4 helpers (column-major) ---
  function perspective(fovy, aspect, near, far) {
    var f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
    return [f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0];
  }
  function mul(a, b) {
    var o = new Array(16);
    for (var i = 0; i < 4; i++) for (var j = 0; j < 4; j++) {
      var s = 0; for (var k = 0; k < 4; k++) s += a[k*4+j] * b[i*4+k];
      o[i*4+j] = s;
    }
    return o;
  }
  function orbitView(dist, yaw, pitch, panX, panY) {
    var cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    // rotate around Z (yaw) then X (pitch), then translate back by dist
    var rz = [cy,sy,0,0, -sy,cy,0,0, 0,0,1,0, 0,0,0,1];
    var rx = [1,0,0,0, 0,cp,sp,0, 0,-sp,cp,0, 0,0,0,1];
    var t  = [1,0,0,0, 0,1,0,0, 0,0,1,0, panX,panY,-dist,1];
    return mul(t, mul(rx, rz));
  }

  function Viewer(canvas, opts) {
    opts = opts || {};
    var gl = canvas.getContext('webgl', {antialias: true, alpha: true});
    if (!gl) throw new Error('WebGL unavailable');
    this.gl = gl; this.canvas = canvas;
    this.pointSize = opts.pointSize || 1.0;   // ~1 => about 1px at default dist
    this.count = 0; this.nSets = 0; this.mix = 0;
    this.yaw = opts.yaw !== undefined ? opts.yaw : -0.6;
    this.pitch = opts.pitch !== undefined ? opts.pitch : -1.05;
    this.dist = opts.dist || 2.6;
    this.panX = 0; this.panY = 0;
    this.bg = opts.background || null;

    var p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VS));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    gl.useProgram(p);
    this.prog = p;
    this.loc = {
      pos: gl.getAttribLocation(p, 'aPos'),
      col: gl.getAttribLocation(p, 'aCol'),
      col2: gl.getAttribLocation(p, 'aCol2'),
      nrm: gl.getAttribLocation(p, 'aNrm'),
      mvp: gl.getUniformLocation(p, 'uMVP'),
      size: gl.getUniformLocation(p, 'uSize'),
      mixv: gl.getUniformLocation(p, 'uMix'),
      gamma: gl.getUniformLocation(p, 'uGamma'),
      light: gl.getUniformLocation(p, 'uLight'),
      shade: gl.getUniformLocation(p, 'uShade')
    };
    // RGB scans of indoor scenes are dark; the feature field already is vivid
    this.gamma = opts.gamma || [1.0, 1.0];
    this.shade = opts.shade !== undefined ? opts.shade : 1.0;
    this.light = opts.light || [0.4, 0.35, 0.85];

    var mp = gl.createProgram();
    gl.attachShader(mp, compile(gl, gl.VERTEX_SHADER, MVS));
    gl.attachShader(mp, compile(gl, gl.FRAGMENT_SHADER, MFS));
    gl.linkProgram(mp);
    if (!gl.getProgramParameter(mp, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(mp));
    this.mProg = mp;
    this.mLoc = {
      bar: gl.getAttribLocation(mp, 'aBar'),
      mvp: gl.getUniformLocation(mp, 'uMVP'),
      query: gl.getUniformLocation(mp, 'uQuery'),
      px: gl.getUniformLocation(mp, 'uPx'),
      thick: gl.getUniformLocation(mp, 'uThick'),
      col: gl.getUniformLocation(mp, 'uCol')
    };
    this.mLen = opts.markerLen || 17;          // arm half-length, px
    this.mThick = opts.markerThick || 1.6;     // half-thickness, px
    this.mBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.mBuf);
    gl.bufferData(gl.ARRAY_BUFFER, crossGeom(this.mLen), gl.STATIC_DRAW);
    this.query = null;
    this.markerAlpha = 0;                      // 0 = hidden
    this.markerColor = opts.markerColor || [0.85, 0.11, 0.11];

    gl.enable(gl.DEPTH_TEST);
    this._bindInput();
    this._raf = null;
  }

  Viewer.prototype.load = function (buffer) {
    var gl = this.gl, dv = new DataView(buffer);
    if (String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3)) !== 'GCPC')
      throw new Error('bad magic');
    var n = dv.getUint32(4, true), sets = dv.getUint32(8, true), flags = dv.getUint32(12, true);
    var off = 16;                                   // keeps xyz 4-byte aligned
    var xyz = new Float32Array(buffer, off, n * 3); off += n * 12;
    var colBufs = [];
    for (var i = 0; i < sets; i++) {
      colBufs.push(new Uint8Array(buffer, off, n * 3)); off += n * 3;
    }
    var nrmArr = null;
    if (flags & 1) { nrmArr = new Int8Array(buffer, off, n * 3); off += n * 3; }
    this.query = null;
    if (flags & 2) {
      this.query = [dv.getFloat32(off, true), dv.getFloat32(off + 4, true),
                    dv.getFloat32(off + 8, true)];
      off += 12;
    }
    if (off !== buffer.byteLength)
      throw new Error('GCPC size mismatch: parsed ' + off + ' of ' + buffer.byteLength +
                      ' bytes (viewer/file format version out of sync?)');
    this.count = n; this.nSets = sets; this.hasNrm = !!nrmArr;

    this.vPos = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vPos); gl.bufferData(gl.ARRAY_BUFFER, xyz, gl.STATIC_DRAW);
    this.vCol = [];
    for (i = 0; i < sets; i++) {
      var b = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, colBufs[i], gl.STATIC_DRAW);
      this.vCol.push(b);
    }
    if (nrmArr) {
      this.vNrm = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vNrm);
      gl.bufferData(gl.ARRAY_BUFFER, nrmArr, gl.STATIC_DRAW);
    } else if (this.vNrm) {
      gl.deleteBuffer(this.vNrm); this.vNrm = null;   // don't reuse stale normals on reload
    }
    this.setPair(0, Math.min(1, sets - 1));
    this.render();
    return this;
  };

  Viewer.prototype.setPair = function (a, b) { this.ia = a; this.ib = b; };
  Viewer.prototype.setMix = function (m) { this.mix = Math.max(0, Math.min(1, m)); this.render(); };
  Viewer.prototype.setMarker = function (a) {
    this.markerAlpha = Math.max(0, Math.min(1, a)); this.render();
  };

  Viewer.prototype.render = function () {
    var gl = this.gl, c = this.canvas;
    var dpr = Math.min(global.devicePixelRatio || 1, 2);
    var w = Math.round(c.clientWidth * dpr), h = Math.round(c.clientHeight * dpr);
    if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
    gl.viewport(0, 0, c.width, c.height);
    if (this.bg) { gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1); }
    else { gl.clearColor(0, 0, 0, 0); }
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (!this.count) return;

    var proj = perspective(50 * Math.PI / 180, c.width / c.height, 0.01, 100);
    var view = orbitView(this.dist, this.yaw, this.pitch, this.panX, this.panY);
    var mvp = new Float32Array(mul(proj, view));
    gl.useProgram(this.prog);
    gl.uniformMatrix4fv(this.loc.mvp, false, mvp);
    // gl_PointSize = uSize / w, and w ~= dist, so uSize ~= wantedPx * dist
    gl.uniform1f(this.loc.size, this.pointSize * (c.height / 300));
    gl.uniform1f(this.loc.mixv, this.mix);
    gl.uniform2f(this.loc.gamma, this.gamma[0], this.gamma[1]);
    gl.uniform3f(this.loc.light, this.light[0], this.light[1], this.light[2]);
    gl.uniform1f(this.loc.shade, this.hasNrm ? this.shade : 0.0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vPos);
    gl.enableVertexAttribArray(this.loc.pos);
    gl.vertexAttribPointer(this.loc.pos, 3, gl.FLOAT, false, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vCol[this.ia]);
    gl.enableVertexAttribArray(this.loc.col);
    gl.vertexAttribPointer(this.loc.col, 3, gl.UNSIGNED_BYTE, true, 0, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.vCol[this.ib]);
    gl.enableVertexAttribArray(this.loc.col2);
    gl.vertexAttribPointer(this.loc.col2, 3, gl.UNSIGNED_BYTE, true, 0, 0);

    if (this.vNrm) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vNrm);
      gl.enableVertexAttribArray(this.loc.nrm);
      gl.vertexAttribPointer(this.loc.nrm, 3, gl.BYTE, true, 0, 0);
    } else if (this.loc.nrm >= 0) {
      gl.disableVertexAttribArray(this.loc.nrm);
      gl.vertexAttrib3f(this.loc.nrm, 0.0, 0.0, 1.0);
    }
    gl.drawArrays(gl.POINTS, 0, this.count);

    if (this.query && this.markerAlpha > 0) {
      var a = this.markerAlpha, m = this.mLoc;
      gl.useProgram(this.mProg);
      gl.uniformMatrix4fv(m.mvp, false, mvp);
      gl.uniform3f(m.query, this.query[0], this.query[1], this.query[2]);
      gl.uniform2f(m.px, 2 / c.width, 2 / c.height);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.mBuf);
      gl.enableVertexAttribArray(m.bar);
      gl.vertexAttribPointer(m.bar, 3, gl.FLOAT, false, 0, 0);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.disable(gl.DEPTH_TEST);        // the query must stay visible inside the cloud
      // white halo first so the cross reads over both bright and dark points
      gl.uniform1f(m.thick, this.mThick * 2.3);
      gl.uniform4f(m.col, 1, 1, 1, 0.85 * a);
      gl.drawArrays(gl.TRIANGLES, 0, 12);
      gl.uniform1f(m.thick, this.mThick);
      gl.uniform4f(m.col, this.markerColor[0], this.markerColor[1], this.markerColor[2], a);
      gl.drawArrays(gl.TRIANGLES, 0, 12);
      gl.enable(gl.DEPTH_TEST);
      gl.disable(gl.BLEND);
      gl.disableVertexAttribArray(m.bar);
    }
  };

  Viewer.prototype._schedule = function () {
    var self = this;
    if (this._raf) return;
    this._raf = requestAnimationFrame(function () { self._raf = null; self.render(); });
  };

  Viewer.prototype._bindInput = function () {
    var self = this, c = this.canvas, drag = null;
    function pos(e) {
      var t = e.touches ? e.touches[0] : e;
      return {x: t.clientX, y: t.clientY};
    }
    function down(e) { drag = pos(e); drag.shift = e.shiftKey; }
    function move(e) {
      if (!drag) return;
      var p = pos(e), dx = p.x - drag.x, dy = p.y - drag.y;
      if (drag.shift) { self.panX += dx * 0.004; self.panY -= dy * 0.004; }
      else {
        self.yaw += dx * 0.008;
        self.pitch = Math.max(-Math.PI, Math.min(0.2, self.pitch + dy * 0.008));
      }
      drag = pos(e); drag.shift = e.shiftKey;
      self._schedule();
      if (e.cancelable) e.preventDefault();
    }
    function up() { drag = null; }
    c.addEventListener('mousedown', down);
    global.addEventListener('mousemove', move);
    global.addEventListener('mouseup', up);
    c.addEventListener('touchstart', down, {passive: true});
    c.addEventListener('touchmove', move, {passive: false});
    c.addEventListener('touchend', up);
    c.addEventListener('wheel', function (e) {
      self.dist = Math.max(0.6, Math.min(8, self.dist * (1 + Math.sign(e.deltaY) * 0.09)));
      self._schedule(); e.preventDefault();
    }, {passive: false});
    global.addEventListener('resize', function () { self._schedule(); });
  };

  Viewer.prototype.spin = function (speed) {
    var self = this;
    speed = speed || 0.0016;
    (function loop() {
      if (self._stop) return;
      self.yaw += speed; self.render();
      requestAnimationFrame(loop);
    })();
    return this;
  };

  global.PCViewer = Viewer;
})(window);
