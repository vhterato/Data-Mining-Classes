&#x09;Aula 6, tema: Aprendizado semi-supervisionado e grafos





&#x09;Nas aulas anteriores, consideramos cenários em que os dados eram rotulados. No caso em que o conjunto era parcialmente rotulado, sempre há a opção de treinar um modelo na parte rotulada e, então, aplicá-lo no restante. No entanto, em casos pouco rotulados, esse procedimento pode ser insuficiente e levar a sobreajuste. Desse modo, há de pensar em como lidar com cenários onde a fração de dados rotulados é muito inferior à de não rotulados.



&#x09;(página 5) É necessário que os dados tenham algum padrão a ser detectado, se não, o modelo não terá sucesso. Ele não se garante sozinho.

&#x09;Os rótulos têm uma estrutura real, não sendo arbitrários.

&#x09;E instâncias vizinhas têm rótulos vizinhos.

&#x09;E, portanto, definimos a similaridade gaussiana entre dois pontos u e v. Há um exemplo visual do queremos fazer na página 6.



&#x09;(página 7) a equação significa que pontos similares têm labels distoantes quanto maior o custo.

&#x09;(página 8) A equação diz que o valor aumenta conforme viola-se o nosso conhecimento sobre o rótulo, lembrando que a soma só vale aos pontos com rótulos

&#x09;(página 9) a função H é a soma do "smoothness" do H\_local sobre um "Graphen" e o H\_global a "acordância" com os rótulos, enquanto o parâmetro u diz quão importante o H\_global é para contrabalancear o H\_local. Enquanto isso, a equação de minimização encontra a classe s com o maior valor (argmax)

