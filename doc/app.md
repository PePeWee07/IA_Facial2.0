
## Explicación Técnica:

- best_similarity = -1:

        Se usa para almacenar el valor máximo de similitud encontrado entre el encoding del rostro actual y cada uno de los encodings almacenados en la base de datos. Se inicializa en -1 porque la similitud del coseno puede variar de -1 a 1, y al comenzar con -1, cualquier comparación real (normalmente positiva) será mayor, lo que permite actualizar esta variable con el valor real de la similitud más alta encontrada.

- best_match = None:

        Se utiliza para guardar el documento (registro) de la base de datos que corresponde al encoding que tiene la mayor similitud con el rostro actual. Se inicia en None para indicar que, hasta el momento de la comparación, no se ha encontrado ningún registro que coincida. A medida que se recorre la base de datos, si se encuentra un registro con una similitud mayor que el valor actual en best_similarity, se actualiza best_match para referenciar ese registro.


### Las dos variables cumplen funciones complementarias, no redundantes:

- best_similarity guarda el valor numérico de la similitud (por ejemplo, 0.95) entre el encoding del rostro actual y uno de los encodings almacenados en la base de datos. Es un número que te dice qué tan parecido es el rostro comparado con otro.

- best_match almacena el documento (registro) de la base de datos que corresponde a ese valor de similitud más alto. Es decir, contiene toda la información (como el nombre, ID, etc.) del usuario cuyo rostro es el más parecido.

        Usarlas juntas te permite no solo conocer el valor de la similitud, sino también saber a quién pertenece ese encoding. Si solo tuvieras una variable, podrías saber la similitud pero no a qué registro se corresponde, o viceversa.