/**
 * Seção 15 - Regressão Linear Multipla
 * Aula 113
 *
 * - O algoritmo trabalha com multiplos INPUTS{_seed} e um unico OUTPUT(_pattern)
 * - O calculo pode conter N INPUTS( colunas distintas ) com varias informações
 * - Ele vai entender qur oa valores de _patternn são as respostaas de _seed
 * - Logo com os proximo dados de entrada input, ele tentara responder se baseando no treino de
 *   _seed e _pattern, tentando chegar a uma conclusao usando a mesma lógica
 *
 * - Um cenário que poderia ser é o calculo do preço de um imovel se baseando em N fatores númericos
 *   e ao final o seu preço como resposta que seria o mesmo que o _pattern
 */

import tf from '@tensorflow/tfjs-node'

const executar = async ({ _seed, _pattern, _input, _epochs }) => {
  const model = tf.sequential()

  const layerConfig = {
    units: _pattern[0].length || 1,
    inputShape: [_pattern.length],
  }
  const seedConfig = [_seed.length, _seed[0].length || 1]
  const inputConfig = [_input.length, _input[0].length || 1]

  model.add(tf.layers.dense(layerConfig))

  model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

  const x = tf.tensor(_seed, seedConfig)
  const y = tf.tensor(_pattern)
  const input = tf.tensor(_input, inputConfig)

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
  // Multiplos inputs com dados não caoticos
  /*
     Input 1
     |  Input 2 
     |  |  Input 3
     |  |  |
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],

  */
  _seed: [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
  ],
  _pattern: [
    /*
     Output1 = Input 1 + Input 2
     |    Output2 = Input2 + Input3
     |    |  
    [3,   5],
    [9,  11],
    [15, 17],
    */
    [3, 5],
    [9, 11],
    [15, 17],
  ], // pattern[i] = seed[i+0] + seed[i+1] + seed[i+N]`
  _input: [[10, 11, 12]],
  _epochs: 500,
})
