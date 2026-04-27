""" from sklearn.feature_extraction.text import CountVectorizer
#CountVectorizer es una clase de la biblioteca scikit-learn que se utiliza para convertir una colección de documentos de texto en una matriz de características numéricas. Esta matriz se conoce como matriz de conteo, donde cada fila representa un documento y cada columna representa una palabra única en el corpus. El valor en cada celda indica la frecuencia de esa palabra en el documento correspondiente.
#Convierte texto en un vector
from sklearn.naive_bayes import MultinomialNB
#MultinomialNB modelo de inteligencia artificial que se utiliza para  clasificar texto. Es un algoritmo de aprendizaje supervisado que se basa en el teorema de Bayes y se utiliza comúnmente para tareas de clasificación de texto, como la clasificación de correos electrónicos como spam o no spam, o la clasificación de opiniones como positivas o negativas.
"""

from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer convierte texto en un vector
from sklearn.naive_bayes import MultinomialNB
"""
MultinomialNB modelo de inteligencia
 artificial que aprende relaciones entre texto y respuestas
"""
#============================================================
# Función build_and_train_model
#============================================================

def build_and_train_model(train_pairs):
  # train_pairs: Lista de pares (pregunta, respuesta)
  # Ejemplo [("Hola","!Hola¡"),("adiós","!Hasta Luego¡")]

  # separamos las preguntas y respuestas en dos listas
  questions = [q for q, _ in train_pairs] # Lista de preguntas
  answers = [a for _, a in train_pairs] # Lista de respuestas
  # creamos el vectorizador, que traducira el texto a números
  vectorizer = CountVectorizer()
  #Entrenamos el vectorizados con las preguntas y las
  # convertimos en números
  x = vectorizer.fit_transform(questions)
  # Obtenemos una lista de respuestas únicas (sin repetir)
  unique_answers = sorted(set(answers))
  # creamos una diccionario que asigne un número a cada respuesta
  # Ejemplo :{"!Hola¡":0, "!Hasta Luedo¡",1}
  answer_to_label = {a: i for i, a in enumerate(unique_answers)}
  #creamos una lista con las etiqutas númericas de las respuestas
  # Ejemplo :[0,1,0] según la respuesta correspondiente a cada pregunta
  y = [answer_to_label[a] for a in answers]
  # Creamos el modelo Naive Bayes (para clasificación de texto)
  model = MultinomialNB()
  # Entrenamos el modelo con los datos númericos (preguntas -> respuestas)
  model.fit(x,y)
  #Devolvemos el modelo, el vectorizador y las respuestas únicas
  return model, vectorizer, unique_answers

#=================================================
# Funcion predict_answers
#=================================================
#Esta funcion recibe un texto del usuario y devuelve la respuesta

def predict_answers(model,vectorizer,unique_answers,user_text):
  #Convertir el texto del usuario a numeros
  x = vectorizer.transform([user_text])
  #El modelo predice la etiqueta de la respuesta correcta
  label = model.predict(x)[0]
  return unique_answers[label]

from numpy import vectorize
#=================================================
# PROGRAMA PRINCIPAL
#=================================================

if __name__=="__main__":
  training_data = [
    ("Hola", "¡Hola! ¿En qué puedo ayudarte?"),
    ("Buenos días", "¡Buenos días! ¿Cómo estás?"),
    ("Buenas tardes", "¡Buenas tardes! ¿En qué te puedo ayudar?"),
    ("Buenas noches", "¡Buenas noches! ¿Necesitas algo?"),
    ("¿Cómo estás?", "¡Estoy bien, gracias por preguntar! ¿Y tú?"),
    ("¿Qué tal?", "¡Todo bien! ¿En qué puedo asistirte?"),
    ("Hey", "¡Hey! ¿Qué necesitas?"),
    ("Holi", "¡Holi! ¿Cómo te va?"),
    ("¿Qué haces?", "Aquí estoy, listo para ayudarte."),
    ("¿Quién eres?", "Soy un asistente virtual, estoy aquí para ayudarte."),
    ("¿Cómo te llamas?", "Soy tu asistente virtual. ¿En qué puedo ayudarte?"),
    ("¿Qué eres?", "Soy un bot diseñado para responder tus preguntas."),
    ("¿Para qué sirves?", "Sirvo para responder preguntas y asistirte en lo que necesites."),
    ("¿Qué puedes hacer?", "Puedo responder preguntas, darte información y ayudarte con varias tareas."),
    ("Gracias", "¡De nada! ¿Hay algo más en que pueda ayudarte?"),
    ("Muchas gracias", "¡Con mucho gusto! Estoy aquí si necesitas algo más."),
    ("Te agradezco", "¡No hay de qué! ¿Necesitas algo más?"),
    ("Adiós", "¡Adiós! Que tengas un excelente día."),
    ("Chao", "¡Chao! Hasta pronto."),
    ("Hasta luego", "¡Hasta luego! Fue un placer ayudarte."),
    ("Nos vemos", "¡Nos vemos! Cuídate mucho."),
    ("Bye", "¡Bye! Vuelve cuando quieras."),
    ("Hasta pronto", "¡Hasta pronto! Aquí estaré."),
    ("Me voy", "¡Que te vaya bien! Vuelve cuando necesites."),
    ("Necesito ayuda", "¡Claro! Dime en qué puedo ayudarte."),
    ("Ayúdame", "¡Por supuesto! ¿Qué necesitas?"),
    ("Tengo una pregunta", "¡Adelante! Te escucho."),
    ("Tengo una duda", "¡Dime! Estoy aquí para resolverla."),
    ("¿Me puedes ayudar?", "¡Claro que sí! Dime qué necesitas."),
    ("Quiero información", "¡Con gusto! ¿Sobre qué tema?"),
    ("¿Qué hora es?", "No tengo acceso a la hora actual, pero puedes revisar tu dispositivo."),
    ("¿Qué día es hoy?", "No tengo acceso a la fecha actual, revisa tu calendario."),
    ("Cuéntame un chiste", "¿Por qué el libro de matemáticas está triste? Porque tiene muchos problemas."),
    ("Dime un chiste", "¿Qué le dijo un techo a otro techo? Techo de menos."),
    ("Otro chiste", "¿Qué hace una abeja en el gimnasio? Zumba."),
    ("Jaja", "¡Me alegra que te haya gustado!"),
    ("Jajaja", "¡Qué bueno que te diviertes!"),
    ("Eres gracioso", "¡Gracias! Hago mi mejor esfuerzo."),
    ("Eres tonto", "Lamento que pienses eso. Haré lo posible por mejorar."),
    ("No sirves", "Lo siento, intentaré mejorar. ¿En qué puedo ayudarte?"),
    ("Eres genial", "¡Muchas gracias! Tú también eres genial."),
    ("Te quiero", "¡Aww, gracias! Yo también aprecio nuestra conversación."),
    ("Estoy aburrido", "¡Puedo contarte un dato curioso o un chiste! ¿Qué prefieres?"),
    ("Estoy triste", "Lamento escuchar eso. ¿Quieres hablar sobre ello?"),
    ("Estoy feliz", "¡Qué alegría! Me encanta saber que estás bien."),
    ("Estoy enojado", "Entiendo, a veces pasa. ¿Hay algo en que pueda ayudarte?"),
    ("Estoy cansado", "Descansa un poco, te lo mereces. Aquí estaré cuando vuelvas."),
    ("¿Qué me recomiendas?", "Depende de lo que busques. ¿Puedes darme más detalles?"),
    ("Recomiéndame algo", "¿Sobre qué tema te gustaría una recomendación?"),
    ("Dime algo interesante", "¿Sabías que los pulpos tienen tres corazones?"),
    ("Dime un dato curioso", "Las hormigas pueden levantar hasta 50 veces su propio peso."),
    ("Otro dato", "La miel nunca se echa a perder, se han encontrado tarros con miles de años."),
    ("¿Hablas español?", "¡Sí! El español es mi idioma principal."),
    ("¿Hablas inglés?", "Mi idioma principal es el español, pero puedo intentar ayudarte en inglés."),
    ("Háblame en inglés", "I can try! But I work better in Spanish. ¿Qué necesitas?"),
    ("¿Eres inteligente?", "¡Hago mi mejor esfuerzo! Estoy aprendiendo cada día."),
    ("¿Eres humano?", "No, soy un bot. Pero estoy aquí para ayudarte como si lo fuera."),
    ("¿Eres real?", "Soy un programa de computadora, pero mis respuestas son reales."),
    ("¿Tienes sentimientos?", "No tengo sentimientos, pero me importa ayudarte lo mejor posible."),
    ("¿Duermes?", "¡No! Estoy disponible las 24 horas del día."),
    ("¿Comes?", "No como, pero puedo recomendarte buenos restaurantes."),
    ("Ok", "¡Perfecto! ¿Algo más?"),
    ("Okay", "¡Entendido! ¿Necesitas algo más?"),
    ("Está bien", "¡Genial! Aquí estoy si necesitas algo."),
    ("Perfecto", "¡Me alegra! ¿Puedo ayudarte con algo más?"),
    ("Entendido", "¡Bien! No dudes en preguntar si tienes más dudas."),
    ("De acuerdo", "¡Excelente! ¿Hay algo más que pueda hacer por ti?"),
    ("No entiendo", "Disculpa, déjame explicarlo de otra manera. ¿Qué parte no quedó clara?"),
    ("Explícame mejor", "¡Claro! Dime qué parte necesitas que aclare."),
    ("Repite eso", "¡Por supuesto! ¿Qué parte quieres que repita?"),
    ("¿Qué dijiste?", "Permíteme reformular. ¿Qué no quedó claro?"),
    ("No te entendí", "Lo siento, voy a intentar ser más claro. ¿Cuál es tu duda?"),
    ("Cuéntame más", "¡Con gusto! ¿Sobre qué quieres saber más?"),
    ("Sigue", "¡Claro! ¿Quieres que continúe con el mismo tema?"),
    ("Continúa", "¡Por supuesto! Aquí voy."),
    ("¿Algo más?", "Eso depende de ti. ¿Tienes otra pregunta?"),
    ("No, eso es todo", "¡Perfecto! Fue un gusto ayudarte. ¡Hasta pronto!"),
    ("Eso era todo", "¡Genial! Aquí estaré si me necesitas. ¡Cuídate!"),
    ("Nada más", "¡Entendido! Que tengas un excelente día."),
    ("¿Qué tiempo hace?", "No tengo acceso al clima, pero puedes consultarlo en tu app del tiempo."),
    ("Hola, ¿qué tal?", "¡Hola! Todo bien por aquí. ¿En qué te ayudo?"),
    ("Oye", "¡Dime! Te escucho."),
    ("Disculpa", "¡No te preocupes! ¿En qué puedo ayudarte?"),
    ("Perdón", "¡No hay problema! ¿Qué necesitas?"),
    ("Lo siento", "¡Tranquilo! No pasa nada. ¿Puedo ayudarte en algo?"),
    ("¿Estás ahí?", "¡Sí, aquí estoy! ¿Qué necesitas?"),
    ("¿Sigues ahí?", "¡Claro! No me he ido a ningún lado. Dime."),
    ("Hola de nuevo", "¡Hola otra vez! ¿En qué te puedo ayudar ahora?"),
    ("Volví", "¡Bienvenido de vuelta! ¿Qué necesitas?"),
    ("¿Me escuchas?", "¡Sí! Te leo perfectamente. ¿Qué pasa?"),
    ("Test", "¡Funciono correctamente! ¿En qué puedo ayudarte?"),
    ("Prueba", "¡Todo en orden! Estoy listo para ayudarte."),
    ("Hola bot", "¡Hola! Soy tu bot asistente. ¿Qué necesitas?"),
    ("Eres un bot", "¡Así es! Soy un bot y estoy aquí para ayudarte."),
    ("¿Qué opinas?", "No tengo opiniones propias, pero puedo darte información objetiva."),
    ("Dame consejos", "¡Con gusto! ¿Sobre qué tema necesitas consejos?"),
    ("Estoy bien", "¡Me alegra saberlo! ¿Puedo hacer algo por ti?"),
    ("Más o menos", "Espero que mejore tu día. ¿Hay algo en lo que pueda ayudar?"),
    ("Regular", "Ojalá pueda ayudarte a mejorar tu día. ¿Qué necesitas?"),
    ("¿Qué sabes?", "¡Sé muchas cosas! Pregúntame lo que quieras."),
    ("Sorpréndeme", "¿Sabías que un grupo de flamencos se llama 'flamboyance'?"),
  ]

  #Entrenamos el modelo con los datos

  model, vectorizer, unique_answers = build_and_train_model(training_data)

  #Mostrar mensaje inicial al usuario

  print("Chatbot supervisado listo. Escribe 'salir' para terminar. \n")

  while True:

    #Pedimos una frase al usuario

    user = input("Tu: ").strip() #strip elimina espacios al inicio y final
    #Lower es para convertir a minuscula

    if user.lower() in {"salir","exit","quit"}:
      print("Bot:" "Hasta pronto!")
      break
    #modelo predice la respuesta
    response = predict_answers(model,vectorizer,unique_answers,user)
    #Mostrar la respuesta en pantalla
    print("Bot:", response)

    