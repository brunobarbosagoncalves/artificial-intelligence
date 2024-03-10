/**
 * Seção 14 - Regressão Linear Simples
 * Aula 99
 */

import tf from '@tensorflow/tfjs-node'

const executar = ({ seed, pattern }) => {
  const tensorX = tf.tensor(seed)
  const tensorY = tf.tensor(pattern)

  const data = []
  const vetorX = TensorToArray(tensorX)
  const vetory = TensorToArray(tensorY)
  const tamX = vetorX.length
  const tamY = vetory.length
  const tempX = vetorX.slice(0, tamY)
  const tempY = vetory
  const dif = tamX - tamY
  if (dif > 0) {
    const regressao = []
    for (let i = 0; i < dif; i++) {
      const temp = RegressaoLinear(tempX, tempY, vetorX[tamY + i])

      regressao.push(temp)

      const novoY = tempY.concat(regressao)
      const tensorZ = tf.tensor(novoY)

      data.push({
        step: `Passo (${i + 1}-${dif})`,
        seed: TensorToString(tensorX),
        pattern: TensorToString(tensorY),
        predict: TensorToString(tensorZ),
      })
    }
  }

  exibir(data)
}

const TensorToArray = (tensor) =>
  JSON.parse(tensor.toString().split('\n')[1].trim())

const TensorToString = (tensor) => tensor.toString().split('\n')[1].trim()

const ArrayToTensor = (array) => tf.tensor(array)

const RegressaoLinear = (arrX, arrY, p) => {
  const dataX = ArrayToTensor(arrX)
  const dataY = ArrayToTensor(arrY)

  // Regra fixa da função regressão linear simples
  const result1 = dataX.sum().mul(dataY.sum()).div(dataX.size)
  const result2 = dataX.sum().mul(dataX.sum()).div(dataX.size)
  const result3 = dataX.mul(dataY).sum().sub(result1)
  const result4 = result3.div(dataX.square().sum().sub(result2))
  const result5 = dataY.mean().sub(result4.mul(dataX.mean()))
  const tensor = result4.mul(p).add(result5)
  const dataZ = TensorToArray(tensor)

  return dataZ
}

const exibir = (list = []) => {
  const result = ['Regressão Linear Simples']

  list.forEach((data) => {
    result.push(`\nPasso: ${data.step}`)
    result.push(`Seed:    ${data.seed}`)
    result.push(`Pattern: ${data.pattern}`)
    result.push(`Predict: ${data.predict}`)
  })

  console.log(result.join('\n'))
}

executar({
  seed: [1, 2, 3, 4, 5, 6, 7, 8, 9],
  pattern: [4, 7, 10, 13], // pattern[i] = seed[i] + seed[i] + seed[i+1]
})

executar({
  seed: [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9],
  pattern: [0.9, 1.8, 2.7, 3.6], // pattern[i] = seed[i] + seed[i+1]
})
