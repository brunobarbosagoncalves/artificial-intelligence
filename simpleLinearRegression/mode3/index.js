/**
 * Seção 14 - Regressão Linear Simples
 * Aula 108
 */

import tf from '@tensorflow/tfjs-node'

const executar = async ({ _seed, _pattern, _input, _epochs }) => {
  const model = tf.sequential()

  model.add(
    tf.layers.dense({
      units: _seed[0].length || 1,
      inputShape: [_seed[0].length || 1],
    }),
  )

  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

  const x = tf.tensor(_seed, [_seed.length, _seed[0].length || 1])
  const y = tf.tensor(_pattern)
  const input = tf.tensor(_input, [_input.length, _input[0].length || 1])

  await model.fit(x, y, { epochs: _epochs })

  const output = model
    .predict(input)
    .dataSync()
    .reduce((a, c) => [...a, Math.ceil(Number(c))], [])

  const predict = tf.tensor(output)

  const result = {
    x: TensorToArray(x),
    y: TensorToArray(y),
    input: TensorToArray(input),
    predict: TensorToArray(predict),
  }

  console.log(JSON.stringify(result, null, 2))
}

const TensorToArray = (tensor) =>
  tensor
    .toString()
    .replaceAll(/(Tensor|\n| )/gim, '')
    .trim()

executar({
  _seed: [1, 2, 3, 4],
  _pattern: [11, 22, 33, 44], // pattern[i] = seed[i] * 10 + seed[i]`
  _input: [20, 30, 40, 50, 60, 70, 80, 90, 100],
  _epochs: 1200,
})
