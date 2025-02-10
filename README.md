# 📄 CRUD con ChromaDB y Streamlit

## 📌 Descripción
Este proyecto es una aplicación basada en **Streamlit** que permite realizar operaciones CRUD sobre una base de datos vectorial implementada con **ChromaDB**. La aplicación utiliza **SentenceTransformers** para generar embeddings a partir de texto, permitiendo realizar consultas semánticas avanzadas.

## 🚀 Tecnologías Utilizadas
- **Python 3.x**
- **Streamlit** (para la interfaz de usuario)
- **ChromaDB** (base de datos de vectores)
- **SentenceTransformers** (modelo de embeddings `all-MiniLM-L6-v2`)
- **Matplotlib** (para visualización de resultados)

## 🎯 Funcionalidades
- **Crear documentos** con embeddings generados automáticamente.
- **Leer documentos** por su ID.
- **Actualizar documentos** existentes.
- **Eliminar documentos** de la base de datos.
- **Realizar consultas avanzadas** con filtros semánticos.
- **Visualizar resultados** de consultas en gráficos interactivos.
- **Listar todos los documentos** almacenados en la base de datos.

## 📁 Estructura del Proyecto
```
/CRUD-ChromaDB-Streamlit
│── main.py                # Código principal de la aplicación
│── README.md              # Documentación del proyecto
│── requirements.txt
```

## 📖 Instalación y Configuración
### 1️⃣ Clonar el repositorio
```sh
git clone https://github.com/SameuelxD/ProyectoCRUDNotebook
cd ProyectoNotebook
```


### 3️⃣ Instalar dependencias
```sh
pip install -r requirements.txt
```

### 4️⃣ Ejecutar la aplicación
```sh
streamlit run main.py
```

## 🛠️ Uso de la Aplicación
1. **Agregar Documento**: Ingresar un ID, contenido y categoría.
2. **Buscar Documento**: Especificar un ID y visualizar detalles.
3. **Actualizar Documento**: Modificar el contenido y metadatos.
4. **Eliminar Documento**: Ingresar un ID y eliminar la entrada.
5. **Consulta Avanzada**: Buscar documentos con filtros semánticos.
6. **Listar Todos los Documentos**: Visualizar todas las entradas registradas.

## 📊 Visualización de Resultados
La aplicación genera gráficos de barras con **Matplotlib** para representar la relevancia de los documentos en las consultas avanzadas.

## 📌 Consideraciones
- ChromaDB permite almacenar y recuperar documentos usando embeddings semánticos.
- Streamlit proporciona una interfaz intuitiva y fácil de usar.
- SentenceTransformers genera embeddings eficientes para mejorar la precisión de las consultas.

