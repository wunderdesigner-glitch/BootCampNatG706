from chatbot.data import training_data # Importamos los datos de entrenamiento
from chatbot.model import build_and_train_model, load_model, predict_answer # Importamos las funciones del modelo
def chat(model, vectorizer, unique_answers):
    """Inicia el modo conversacion"""
    print("\n 🤖 Chatbot listo. Escribe 'salir' para terminar. \n")
    while True: 
        user = input("Tú: ").strip() 
        if user.lower() in {"salir", "exit", "quit"}: 
            print("Bot: ¡Hasta luego!")
            break
        response = predict_answer(model, vectorizer, unique_answers, user) 
        print("Bot:", response) 

def main():
    model, vectorizer, unique_answers = load_model() 
    #Menu principal
    while True:
        print("\n=== 🤖 MENU PRINCIPAL DE LA CONVERSACION ===")
        print("1️⃣ Chatear con el modelo")
        print("2️⃣ Reentrenar el modelo")
        print("3️⃣ Salir")
        opcion = input("\n Selecciona una opción (1-3); ").strip()
        if opcion == "1":
            if model is None:
                print("\n ⚠️ No hay modelo entrenado. Entrenalo primero.")
            else:
                chat(model, vectorizer,unique_answers)
        elif opcion == "2":
            print("\n 🔄 Reentrenar el modelo con los nuevos datos...")
            model, vectorizer, unique_answers = build_and_train_model(training_data)
            print("\n ✅ Modelo actualizado correctamente.")
        elif opcion == "3":
            print ("\n 👋 ¡Hasta luego!")
            break
        else:
            print("\n ❌ Opción no válida. Intenta de nuevo.")
if __name__ == "__main__":
    main()