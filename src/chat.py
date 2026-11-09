"""CLI de chat: simula um chat no terminal sobre o conteúdo do PDF.

Ao receber uma PERGUNTA: vetoriza, busca k=10 no pgVector, monta o prompt e
chama a LLM (tudo dentro da chain de search_prompt). Digite 'sair' para encerrar.
"""
from providers import PROVIDER
from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print(f"Chat iniciado (provider={PROVIDER}). Digite 'sair' para encerrar.\n")
    print("Faça sua pergunta:\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break

        resposta = chain.invoke(pergunta)
        print(f"RESPOSTA: {resposta}\n")


if __name__ == "__main__":
    main()
