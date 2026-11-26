const SAMPLE_SIZE = 224;

export const INPUT_QUALITY_POLICY = Object.freeze({
  shortestSide: 192,
  longestSide: 256,
  minimumVisibleFraction: .90,
  clippedFraction: .985,
  maximumDarkP90: 28,
  minimumContrast: 10,
  flatEdgeFraction: .004,
  maximumBlurRms: 1.35,
  maximumBlurEdgeFraction: .01,
});

const REQUESTS = {
  resolution: 'Переснимите объект ближе на основную камеру. Прикрепите исходное фото размером от 192 × 256 пикселей.',
  transparent: 'Прикрепите исходную фотографию без прозрачного фона или вырезанных участков.',
  clipped: 'Переснимите объект при рассеянном свете. Уберите прямой блик и настройте экспозицию по поверхности шкафа.',
  dark: 'Подсветите шкаф или переснимите при дневном свете. На кадре должны быть видны контуры двери и покрытие.',
  flat: 'Переснимите шкаф целиком. Оставьте в кадре контуры двери, крепления и поверхность корпуса.',
  blur: 'Зафиксируйте камеру, коснитесь шкафа на экране для фокусировки и переснимите при хорошем свете.',
};

function result(reason, metrics) {
  return { ok: reason === null, reason, request: reason ? REQUESTS[reason] : null, metrics };
}

function percentile(histogram, count, fraction) {
  const rank = Math.max(1, Math.ceil(count * fraction));
  let accumulated = 0;
  for (let i = 0; i < histogram.length; i++) {
    accumulated += histogram[i];
    if (accumulated >= rank) return i;
  }
  return 255;
}

/** Check readability on an already decoded ImageBitmap or HTMLImageElement. */
export function assessInputQuality(bitmap) {
  const width = bitmap.naturalWidth ?? bitmap.width;
  const height = bitmap.naturalHeight ?? bitmap.height;
  const policy = INPUT_QUALITY_POLICY;
  const metrics = { width, height };
  if (Math.min(width, height) < policy.shortestSide || Math.max(width, height) < policy.longestSide) {
    return result('resolution', metrics);
  }

  const scale = SAMPLE_SIZE / Math.max(width, height);
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(3, Math.round(width * scale));
  canvas.height = Math.max(3, Math.round(height * scale));
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  const { data } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const gray = new Uint8Array(canvas.width * canvas.height);
  const histogram = new Uint32Array(256);
  let visible = 0;
  let nearWhite = 0;
  for (let i = 0; i < gray.length; i++) {
    const index = i * 4;
    if (data[index + 3] < 230) continue;
    const value = (54 * data[index] + 183 * data[index + 1] + 19 * data[index + 2] + 128) >> 8;
    visible++;
    gray[i] = value;
    histogram[value]++;
    if (value >= 250) nearWhite++;
  }
  metrics.visibleFraction = visible / gray.length;
  if (metrics.visibleFraction < policy.minimumVisibleFraction) return result('transparent', metrics);

  const low = percentile(histogram, visible, .05);
  metrics.lumaP90 = percentile(histogram, visible, .90);
  metrics.lumaP95 = percentile(histogram, visible, .95);
  metrics.lumaRange = metrics.lumaP95 - low;
  metrics.nearWhiteFraction = nearWhite / visible;
  if (metrics.nearWhiteFraction >= policy.clippedFraction) return result('clipped', metrics);
  if (metrics.lumaP90 <= policy.maximumDarkP90) return result('dark', metrics);

  let count = 0;
  let squaredLaplacian = 0;
  let edges = 0;
  const stride = canvas.width;
  const fullyOpaque = visible === gray.length;
  for (let y = 1; y < canvas.height - 1; y++) {
    for (let x = 1; x < stride - 1; x++) {
      const i = y * stride + x;
      if (!fullyOpaque && (data[i * 4 + 3] < 230 || data[(i - 1) * 4 + 3] < 230 ||
          data[(i + 1) * 4 + 3] < 230 || data[(i - stride) * 4 + 3] < 230 ||
          data[(i + stride) * 4 + 3] < 230)) continue;
      const laplacian = gray[i - 1] + gray[i + 1] + gray[i - stride] + gray[i + stride] - 4 * gray[i];
      squaredLaplacian += laplacian * laplacian;
      if (Math.abs(laplacian) >= 5) edges++;
      count++;
    }
  }
  metrics.laplacianRms = count ? Math.sqrt(squaredLaplacian / count) : 0;
  metrics.edgeFraction = count ? edges / count : 0;
  if (metrics.lumaRange <= policy.minimumContrast && metrics.edgeFraction <= policy.flatEdgeFraction) {
    return result('flat', metrics);
  }
  if (metrics.laplacianRms <= policy.maximumBlurRms && metrics.edgeFraction <= policy.maximumBlurEdgeFraction) {
    return result('blur', metrics);
  }
  return result(null, metrics);
}
