import streamlit as st
from typing import List, Any, Dict, Optional
from sentence_transformers import SentenceTransformer
import chromadb
import matplotlib.pyplot as plt

# Configuración de ChromaDB
chroma_client = chromadb.Client()
collection_name = "vector_db_demo"

if collection_name not in chroma_client.list_collections():
    collection = chroma_client.create_collection(name=collection_name)
else:
    collection = chroma_client.get_collection(name=collection_name)

# Inicialización del modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Estilos CSS personalizados
st.markdown("""
    <style>
    .stButton button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .stButton button:hover {
        background-color: #45a049;
    }
    .stTextInput input, .stTextArea textarea {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .stExpander {
        border-radius: 5px;
        border: 1px solid #ccc;
        padding: 10px;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4CAF50;
    }
    .stDataFrame {
        border-radius: 5px;
        border: 1px solid #ccc;
    }
    </style>
    """, unsafe_allow_html=True)

def generate_embedding(text: str) -> List[float]:
    """Genera un embedding para el texto dado."""
    return model.encode(text).tolist()

def create_entry(id: str, text: str, metadata: Dict[str, Any]) -> bool:
    """Crea una nueva entrada en la base de datos."""
    try:
        existing_entry = collection.get(ids=[id])
        if existing_entry and existing_entry["documents"]:
            st.error(f"Error: El ID '{id}' ya está registrado.")
            return False
        if not id or not text or not metadata:
            st.error("Error: Todos los campos son obligatorios.")
            return False
        embedding = generate_embedding(text)
        collection.add(ids=[id], embeddings=[embedding], metadatas=[metadata], documents=[text])
        st.success(f"Documento '{id}' guardado correctamente.")
        return True
    except Exception as e:
        st.error(f"Error al crear el documento: {e}")
        return False

def read_entry(id: str) -> Optional[Dict[str, Any]]:
    """Lee una entrada de la base de datos por su ID."""
    try:
        result = collection.get(ids=[id])
        if result and result["documents"]:
            return result
        else:
            st.error(f"No se encontró el documento con ID '{id}'.")
            return None
    except Exception as e:
        st.error(f"Error al leer el documento: {e}")
        return None

def update_entry(id: str, new_text: str, new_metadata: Dict[str, Any]) -> bool:
    """Actualiza una entrada existente en la base de datos."""
    try:
        existing_entry = collection.get(ids=[id])
        if not existing_entry or not existing_entry["documents"]:
            st.error(f"Error: No se encontró un documento con ID '{id}'.")
            return False
        if not new_text or not new_metadata:
            st.error("Error: Todos los campos son obligatorios.")
            return False
        embedding = generate_embedding(new_text)
        collection.update(ids=[id], embeddings=[embedding], metadatas=[new_metadata], documents=[new_text])
        st.success(f"Documento '{id}' actualizado correctamente.")
        return True
    except Exception as e:
        st.error(f"Error al actualizar el documento: {e}")
        return False

def delete_entry(id: str) -> bool:
    """Elimina una entrada de la base de datos por su ID."""
    try:
        existing_entry = collection.get(ids=[id])
        if not existing_entry or not existing_entry["documents"]:
            st.error(f"Error: No se encontró un documento con ID '{id}'.")
            return False
        collection.delete(ids=[id])
        st.success(f"Documento '{id}' eliminado correctamente.")
        return True
    except Exception as e:
        st.error(f"Error al eliminar el documento: {e}")
        return False

def query_with_filters(query_text: str, filters: Dict[str, Any], top_k: int = 5) -> Optional[Dict[str, Any]]:
    """Realiza una consulta en la base de datos con filtros."""
    try:
        if not query_text:
            st.error("Error: El texto de la consulta no puede estar vacío.")
            return None
        embedding = generate_embedding(query_text)
        return collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=filters,
            include=["documents", "metadatas", "distances"]  # Incluir distancias
        )
    except Exception as e:
        st.error(f"Error al realizar la consulta: {e}")
        return None

def plot_query_results(results: Dict[str, Any]):
    """Muestra los resultados de la consulta en una gráfica mejorada."""
    if results and results["documents"]:
        # Asegurarse de que documents sea una lista plana de strings
        documents = results["documents"][0] if isinstance(results["documents"][0], list) else results["documents"]
        
        # Calcular la relevancia basada en las distancias (si están disponibles)
        if "distances" in results and results["distances"]:
            scores = [1 - distance for distance in results["distances"][0]]  # Convertir distancias a relevancia
        else:
            scores = [1 for _ in documents]  # Dummy scores si no hay distancias
        
        # Crear la gráfica mejorada
        fig, ax = plt.subplots(figsize=(10, len(documents) * 0.5))  # Ajustar el tamaño según la cantidad de documentos
        ax.barh(documents, scores, color="#4CAF50")  # Usar un color moderno
        ax.set_xlabel("Relevancia", fontsize=12)
        ax.set_title("Resultados de la consulta", fontsize=14)
        ax.set_xlim(0, 1)  # Limitar el eje X a valores entre 0 y 1
        plt.grid(axis="x", linestyle="--", alpha=0.7)  # Agregar una cuadrícula
        st.pyplot(fig)
    else:
        st.warning("No hay resultados para mostrar.")

def list_all_entries():
    """Lista todos los documentos en la base de datos."""
    try:
        all_entries = collection.get()
        if all_entries and all_entries["documents"]:
            st.write("### Lista de Documentos")
            for id, doc, metadata in zip(all_entries["ids"], all_entries["documents"], all_entries["metadatas"]):
                st.write(f"**ID:** {id}")
                st.write(f"**Contenido:** {doc}")
                st.write(f"**Metadatos:** {metadata}")
                st.write("---")
        else:
            st.info("No hay documentos en la base de datos.")
    except Exception as e:
        st.error(f"Error al listar los documentos: {e}")

# Interfaz de Streamlit
st.title("📄 CRUD con ChromaDB y Streamlit")
st.markdown("""
    **Bienvenido a la aplicación de gestión de documentos.**  
    Aquí puedes crear, leer, actualizar y eliminar documentos en una base de datos vectorial.
    """)

# Crear entrada
with st.expander("➕ Agregar Documento", expanded=False):
    id = st.text_input("ID del Documento", key="create_id")
    texto = st.text_area("Contenido del Documento", key="create_text")
    categoria = st.text_input("Categoría", key="create_category")
    if st.button("Guardar Documento", key="create_button"):
        if id and texto and categoria:
            create_entry(id, texto, {"categoria": categoria})
        else:
            st.error("Todos los campos son obligatorios.")

# Leer entrada
with st.expander("🔍 Buscar Documento por ID", expanded=False):
    search_id = st.text_input("ID a buscar", key="search_id")
    if st.button("Buscar", key="search_button"):
        result = read_entry(search_id)
        if result:
            st.write("### Documento Encontrado")
            st.write(f"**ID:** {search_id}")
            st.write(f"**Contenido:** {result['documents'][0]}")
            st.write(f"**Metadatos:** {result['metadatas'][0]}")

# Actualizar entrada
with st.expander("✏️ Actualizar Documento", expanded=False):
    update_id = st.text_input("ID del Documento a actualizar", key="update_id")
    new_text = st.text_area("Nuevo Contenido", key="update_text")
    new_categoria = st.text_input("Nueva Categoría", key="update_category")
    if st.button("Actualizar", key="update_button"):
        if update_id and new_text and new_categoria:
            update_entry(update_id, new_text, {"categoria": new_categoria})
        else:
            st.error("Todos los campos son obligatorios.")

# Eliminar entrada
with st.expander("🗑️ Eliminar Documento", expanded=False):
    delete_id = st.text_input("ID del Documento a eliminar", key="delete_id")
    if st.button("Eliminar", key="delete_button"):
        delete_entry(delete_id)

# Consulta con filtros
# Consulta con filtros
with st.expander("🔎 Consulta Avanzada", expanded=False):
    query_text = st.text_input("Texto para consulta", key="query_text")
    query_categoria = st.text_input("Filtrar por Categoría", key="query_category")
    if st.button("Consultar", key="query_button"):
        filters = {"categoria": query_categoria} if query_categoria else {}
        results = query_with_filters(query_text, filters)
        if results:
            st.write("### Resultados de la Consulta")
            st.write(results)  
            plot_query_results(results)  

# Listar todos los documentos
with st.expander("📂 Listar Todos los Documentos", expanded=False):
    if st.button("Mostrar Todos los Documentos", key="list_button"):
        list_all_entries()