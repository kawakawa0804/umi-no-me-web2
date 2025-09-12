const video  = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');

/* 800×600 でカメラを取得 */
async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: 800, height: 600 }
  });
  return new Promise(res => {
    video.srcObject = stream;
    video.onloadedmetadata = () => { video.play(); res(); };
  });
}

/* 1 秒ごとにフレーム送信 → 推論結果を描画 */
async function sendFrame() {
  /* video の実サイズをそのままキャンバスへ */
  canvas.width  = video.videoWidth;   // 800
  canvas.height = video.videoHeight;  // 600

  /* 現フレームを JPEG にしてサーバーへ POST */
  ctx.drawImage(video, 0, 0);
  const dataURL = canvas.toDataURL('image/jpeg');

  const res = await fetch('/detect', {
    method : 'POST',
    headers: { 'Content-Type': 'application/json' },
    body   : JSON.stringify({ image: dataURL })
  });
  const { detections } = await res.json();

  /* いったんフレームを書き直し、上に枠とラベルを追加 */
  ctx.drawImage(video, 0, 0);

  detections.forEach(d => {
    // 緑枠
    ctx.strokeStyle = '#00FF00';
    ctx.lineWidth   = 2;
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);

    // ラベル背景（半透明）
    const label = `${d.label} ${d.conf}`;
    ctx.font = '18px sans-serif';
    const bgW = ctx.measureText(label).width + 8;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(d.x1, d.y1 - 24, bgW, 24);

    // ラベル文字
    ctx.fillStyle = '#00FF00';
    ctx.fillText(label, d.x1 + 4, d.y1 - 6);
  });
}

/* 初期化 */
(async () => {
  await setupCamera();
  setInterval(sendFrame, 1000);   // ← 必要なら 500 などに変えて FPS↑
})();
