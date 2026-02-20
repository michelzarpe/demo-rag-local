from langchain_chroma import Chroma # trazer o Chroma que é necessário pois ele é usado para criar o banco de dados de vetores e realizar a busca de similaridade
from langchain_ollama import OllamaEmbeddings # trazer o OllamaEmbeddings que é necessário para criar as representações vetoriais dos textos, que serão usadas para a busca de similaridade no banco de dados de vetores.
from langchain_community.llms import Ollama # trazer o Ollama que é necessário para criar o modelo de linguagem que será usado para gerar as respostas com base no contexto encontrado.

CHROMA_PATH = "chroma"

def ask_rag(question: str):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    ) # criar as representações vetoriais dos textos usando o modelo "nomic-embed-text" do OllamaEmbeddings. Essas representações vetoriais serão usadas para a busca de similaridade no banco de dados de vetores. O modelo "nomic-embed-text" é um modelo pré-treinado que pode gerar representações vetoriais eficazes para uma variedade de textos.

    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    docs = db.similarity_search(question, k=4) # realizar a busca de similaridade no banco de dados de vetores usando a pergunta feita como consulta. O método "similarity_search" retorna os documentos mais semelhantes à pergunta, com base nas representações vetoriais criadas pelo OllamaEmbeddings. O parâmetro "k" define o número de documentos mais semelhantes a serem retornados.

    context = "\n\n".join([doc.page_content for doc in docs]) # criar o contexto para a resposta, juntando o conteúdo dos documentos encontrados na busca de similaridade. O conteúdo de cada documento é acessado através do atributo "page_content". Os documentos são separados por duas quebras de linha ("\n\n") para melhorar a legibilidade do contexto.

    prompt = f"""
        Use apenas o contexto abaixo para responder a pergunta.
        Se a resposta não estiver no contexto, diga que não encontrou a informação.

        Contexto:
        {context}

        Pergunta:
        {question}
    """

    llm = Ollama(model="mistral") # criar o modelo de linguagem que será usado para gerar as respostas com base no contexto encontrado. O modelo "mistral" é um modelo de linguagem pré-treinado que pode ser usado para gerar respostas coerentes e relevantes com base no contexto fornecido.

    return llm.invoke(prompt) # usar o modelo de linguagem para gerar a resposta com base no prompt criado, que inclui o contexto encontrado e a pergunta feita. O método "invoke" é usado para chamar o modelo de linguagem e obter a resposta gerada.