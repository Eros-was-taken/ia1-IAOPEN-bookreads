# ia1-IAOPEN-bookreads
Project to make a model that can extract the most probable genre of a book given the description or summary of it
Emanuel Navas 2230958, Sebastián Laguna 2211911, Nicolas Garcia Tamayo 2211233

https://github.com/Eros-was-taken/ia1-IAOPEN-bookreads/blob/master/bookreads_w_descriptions_ds.csv

Problema y relevancia:
El problema que buscamos resolver es la clasificación automática de textos literarios por géneros o categorías temáticas. Esta problemática es relevante porque en la era digital existe una sobrecarga de información textual que requiere organización eficiente. La clasificación manual de textos es un proceso laborioso y subjetivo que limita la capacidad de bibliotecas digitales, plataformas de lectura y sistemas de recomendación para ofrecer contenido personalizado. Una solución automatizada permitiría mejorar la experiencia del usuario y optimizar la gestión de grandes volúmenes de contenido literario.

Objetivo del análisis:

El análisis exploratorio de datos nos permitirá comprender las características intrínsecas de los textos literarios que permiten distinguir entre géneros. Identificaremos patrones en la estructura, vocabulario, longitud y elementos estilísticos que caracterizan cada categoría. Este conocimiento será fundamental para seleccionar las características más discriminativas que alimentarán nuestro modelo de clasificación y determinar qué técnicas de procesamiento de lenguaje natural serán más efectivas.


Métricas o indicadores:
Precisión: de las veces que el modelo dijo “este es el género”, ¿cuántas veces acertó? Importante para no mostrar géneros incorrectos a los usuarios.

Recall: de todos los libros de un género, ¿cuántos detectó el modelo? Sirve para no dejar afuera géneros menos comunes.

F1 (por clase / macro): resumen entre precisión y recall. Bueno cuando hay clases desbalanceadas y queremos una cifra única.

Matriz de confusión: tabla que enseña qué géneros se confunden entre sí. Útil para ver errores típicos y mejorar etiquetas o features.

Top-k accuracy: ¿el género correcto aparece entre las k primeras opciones? Muy práctico para sistemas de recomendación que muestran varias sugerencias.

ROC-AUC (para salidas probabilísticas): indica si el modelo separa bien clases sin depender de un umbral. Bueno para comparar modelos.

Tiempo de inferencia: cuánto tarda una predicción; importante si queremos respuestas rápidas en producción.

Tamaño del modelo / memoria: cuánto pesa el modelo; importante si lo vamos a correr en servidores pequeños o en dispositivos.

Motivación de la elección :
Elegimos este problema por su combinación de NLP aplicado a un dominio culturalmente relevante, la disponibilidad de datos públicos (Goodreads) y su impacto práctico en recomendaciones y gestión bibliotecaria.

Después del notebook :
-Datos utilizados 
 Datasets goodreads_books.csv y goodreads_w_description_ds.csv (tablas CSV): descripciones y metadatos de libros (títulos, autores, géneros/etiquetas, calificaciones, año, etc.).

-Información contenida en los datos:
Las tablas incluyen: título, autor, descripción/sinopsis (texto), géneros/etiquetas (posible multilabel), calificaciones y metadatos (año, idioma, etc.). Para la clasificación son útiles las descripciones (texto libre) y las etiquetas de género; además se pueden derivar características como longitud del texto, frecuencia de palabras, n-grams, POS tags y métricas de legibilidad que ayudan a separar categorías temáticas.

Desafíos asociados a los datos:
descripciones ausentes o muy breves, etiquetas inconsistentes o multilabel, desequilibrio entre géneros (sesgo hacia populares), ruido (HTML, caracteres especiales), lenguaje informal o variado, metadatos incompletos y posible sesgo cultural/idiomático. Estas limitaciones requieren limpieza, imputación o filtrado, re-etiquetado parcial y estrategias de balanceo para evitar modelos sesgados o con pobre generalización.
