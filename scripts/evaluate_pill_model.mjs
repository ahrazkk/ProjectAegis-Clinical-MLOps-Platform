import fs from 'node:fs';
import path from 'node:path';
import http from 'node:http';
import { fileURLToPath } from 'node:url';
import * as tf from '@tensorflow/tfjs';
import sharp from 'sharp';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');

function startStaticServer(rootDir) {
  return new Promise((resolve, reject) => {
    const server = http.createServer((req, res) => {
      try {
        const reqPath = decodeURIComponent((req.url || '/').split('?')[0]);
        const safePath = reqPath.replace(/^\/+/, '');
        const filePath = path.join(rootDir, safePath);
        if (!filePath.startsWith(rootDir)) {
          res.writeHead(403);
          res.end('Forbidden');
          return;
        }

        if (!fs.existsSync(filePath) || fs.statSync(filePath).isDirectory()) {
          res.writeHead(404);
          res.end('Not found');
          return;
        }

        const ext = path.extname(filePath).toLowerCase();
        const contentType =
          ext === '.json' ? 'application/json' :
          ext === '.bin' ? 'application/octet-stream' :
          ext === '.jpg' || ext === '.jpeg' ? 'image/jpeg' :
          ext === '.png' ? 'image/png' :
          'application/octet-stream';

        res.writeHead(200, { 'Content-Type': contentType });
        fs.createReadStream(filePath).pipe(res);
      } catch (err) {
        res.writeHead(500);
        res.end(String(err));
      }
    });

    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address();
      resolve({ server, port: addr.port });
    });
  });
}

async function imageToTensor(imagePath) {
  const { data, info } = await sharp(imagePath)
    .resize(224, 224, { fit: 'cover' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const pixels = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  return tf.tidy(() => {
    const img = tf.tensor3d(pixels, [info.height, info.width, info.channels], 'int32');
    return img.toFloat().div(255.0).expandDims(0);
  });
}

function listImagesRecursively(dir) {
  const out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      out.push(...listImagesRecursively(full));
    } else if (/\.(jpg|jpeg|png)$/i.test(entry.name)) {
      out.push(full);
    }
  }
  return out;
}

function patchKeras3Json(obj) {
  if (!obj || typeof obj !== 'object') return;

  if (Object.prototype.hasOwnProperty.call(obj, 'batch_shape') && !Object.prototype.hasOwnProperty.call(obj, 'batchInputShape')) {
    obj.batchInputShape = obj.batch_shape;
    delete obj.batch_shape;
  }

  for (const key of Object.keys(obj)) {
    const val = obj[key];
    if (Array.isArray(val)) {
      for (const item of val) patchKeras3Json(item);
    } else if (val && typeof val === 'object') {
      patchKeras3Json(val);
    }
  }
}

async function main() {
  const modelDir = path.join(projectRoot, 'public', 'models', 'pill-classifier');
  const modelJsonPath = path.join(modelDir, 'model.json');
  const patchedModelJsonPath = path.join(modelDir, 'model.compat.json');
  const labelsPath = path.join(modelDir, 'labels.json');
  const evalValDir = path.resolve(projectRoot, '..', 'pill_eval_data', 'val');

  if (!fs.existsSync(labelsPath)) {
    throw new Error(`labels.json not found: ${labelsPath}`);
  }
  if (!fs.existsSync(evalValDir)) {
    throw new Error(`Validation directory not found: ${evalValDir}`);
  }

  const labels = JSON.parse(fs.readFileSync(labelsPath, 'utf8')).map((x) => String(x).toLowerCase());
  const labelToIndex = new Map(labels.map((name, i) => [name, i]));

  const classDirs = fs.readdirSync(evalValDir, { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name);

  const candidates = [];
  for (const cls of classDirs) {
    const clsLower = cls.toLowerCase();
    if (!labelToIndex.has(clsLower)) continue;
    const clsPath = path.join(evalValDir, cls);
    for (const imgPath of listImagesRecursively(clsPath)) {
      candidates.push({ imgPath, trueLabel: clsLower });
    }
  }

  if (candidates.length === 0) {
    throw new Error('No evaluable images found (class names did not match labels.json).');
  }

  const { server, port } = await startStaticServer(path.join(projectRoot, 'public'));
  const modelUrl = `http://127.0.0.1:${port}/models/pill-classifier/model.json`;
  const modelCompatUrl = `http://127.0.0.1:${port}/models/pill-classifier/model.compat.json`;

  try {
    console.log(`Loading model from ${modelUrl}`);
    let model;
    try {
      model = await tf.loadLayersModel(modelUrl);
    } catch (err) {
      console.warn(`Primary load failed: ${err?.message || err}`);
      console.warn('Attempting compatibility patch for Keras-3 style model.json...');

      const raw = JSON.parse(fs.readFileSync(modelJsonPath, 'utf8'));
      patchKeras3Json(raw);
      fs.writeFileSync(patchedModelJsonPath, JSON.stringify(raw));
      model = await tf.loadLayersModel(modelCompatUrl);
    }
    console.log(`Model loaded. Evaluating ${candidates.length} images...`);

    let top1 = 0;
    let top3 = 0;
    const perClass = new Map();

    for (const { imgPath, trueLabel } of candidates) {
      const input = await imageToTensor(imgPath);
      const pred = model.predict(input);
      const scores = Array.from(await pred.data());
      input.dispose();
      pred.dispose();

      const ranked = scores
        .map((s, i) => ({ s, i }))
        .sort((a, b) => b.s - a.s)
        .slice(0, 3);

      const top1Label = labels[ranked[0].i];
      const top3Labels = ranked.map((r) => labels[r.i]);
      const hit1 = top1Label === trueLabel;
      const hit3 = top3Labels.includes(trueLabel);
      if (hit1) top1 += 1;
      if (hit3) top3 += 1;

      if (!perClass.has(trueLabel)) perClass.set(trueLabel, { total: 0, top1: 0, top3: 0 });
      const row = perClass.get(trueLabel);
      row.total += 1;
      if (hit1) row.top1 += 1;
      if (hit3) row.top3 += 1;
    }

    const total = candidates.length;
    console.log('----- RESULTS -----');
    console.log(`Total images: ${total}`);
    console.log(`Top-1 accuracy: ${(100 * top1 / total).toFixed(2)}% (${top1}/${total})`);
    console.log(`Top-3 accuracy: ${(100 * top3 / total).toFixed(2)}% (${top3}/${total})`);

    const worst = Array.from(perClass.entries())
      .map(([label, r]) => ({ label, ...r, acc1: r.top1 / r.total }))
      .sort((a, b) => a.acc1 - b.acc1)
      .slice(0, 8);

    console.log('Worst classes (Top-1):');
    for (const w of worst) {
      console.log(`  ${w.label}: ${(100 * w.acc1).toFixed(1)}% (${w.top1}/${w.total})`);
    }
  } finally {
    server.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
