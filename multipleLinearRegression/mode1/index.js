/**
 * Seção 15 - Regressão Linear Multipla
 * Aula 113
 */

import tf from '@tensorflow/tfjs-node'

const executar = async ({ _seed, _pattern, _input, _epochs }) => {
  const model = tf.sequential()

  const inputAndAnswerFormat = [_input.length, _input[0].length || 1]

  model.add(
    tf.layers.dense({
      units: inputAndAnswerFormat[0],
      inputShape: [inputAndAnswerFormat[1]],
    }),
  )

  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

  const x = tf.tensor(_seed, [_seed.length, _seed[0].length || 1])
  const y = tf.tensor(_pattern)
  const input = tf.tensor(_input, inputAndAnswerFormat)

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
  _seed: [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
  ],
  _pattern: [[6], [15], [24]], // pattern[i] = seed[i+0] + seed[i+1] + seed[i+N]`
  _input: [[10, 11, 12]],
  _epochs: 400,
})
