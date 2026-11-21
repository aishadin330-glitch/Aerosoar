import { MLModel } from '../js/sim/ml.js';
import fs from 'node:fs';
for (const n of ['lstm', 'transformer']) {
  const m = new MLModel(JSON.parse(fs.readFileSync(`ml/${n}.json`)));
  console.log(n, 'max |JS - PyTorch| on 5 test windows =', m.selfCheck().toExponential(2));
}
