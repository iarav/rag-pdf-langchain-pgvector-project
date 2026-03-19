from search import search_prompt


def main():
    print("Chat RAG iniciado.")
    print("Digite sua pergunta e pressione Enter.")
    print("Para sair, digite: sair")

    while True:
        question = input("\nPERGUNTA: ").strip()

        if not question:
            print("Digite uma pergunta válida.")
            continue

        if question.lower() in {"sair", "exit", "quit"}:
            print("Encerrando chat.")
            break

        try:
            answer = search_prompt(question)
            
            if not answer:
                print("Desculpe, não consegui encontrar uma resposta para sua pergunta.")
            else:
                print(f"RESPOSTA: {answer}")
        except Exception as exc:
            print(f"Erro ao processar pergunta: {exc}")

if __name__ == "__main__":
    main()