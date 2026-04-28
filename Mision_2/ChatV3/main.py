from chatbot.data import training_data # Importamos los datos de entrenamiento
from chatbot.model import build_and_train_model, load_model, predict_answer # Importamos las funciones del modelo
def main():
    model, vectorizer, unique_answers = load_model() # Intentamos cargar el modelo entrenado
    if model is None: # Si no se pudo cargar el modelo, lo entrenamos
        model, vectorizer, unique_answers = build_and_train_model(training_data) # Entrenamos el modelo con los datos de entrenamiento
    print("\n 🤖 Chatbot listo. Escribe 'salir' para salir.")
    while True: # Bucle infinito para interactuar con el usuario
        user = input("Tú: ").strip() # Leemos la entrada del usuario y eliminamos espacios en blanco
        if user.lower() in {"salir", "exit", "quit"}: # Si el usuario quiere salir, terminamos el programa
            print("Bot: ¡Hasta luego!")
            break
        response = predict_answer(model, vectorizer, unique_answers, user) # Predecimos la respuesta usando el modelo
        print("Bot:", response) # Imprimimos la respuesta del bot
if __name__ == "__main__":
    main() # Ejecutamos la función principal
