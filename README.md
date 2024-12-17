# API-Asistente

El proyecto “Sistema de IA Generativa para la gestión de licitaciones públicas” se
enmarca dentro del programa “Activa Startups” del Plan de Recuperación, Transformación y
Resiliencia, cuyo propósito es fomentar la digitalización de las pymes a través de la
implementación de soluciones innovadoras basadas en tecnologías emergentes. El objetivo del
programa es mejorar la competitividad de las empresas mediante la adopción de herramientas
avanzadas que faciliten la automatización y optimización de procesos críticos. En este contexto,
Mottum Analytica propone una solución basada en un Modelo de Lenguaje (LLM) para abordar las
necesidades específicas de CODEXCA en la gestión de su patrimonio documental y operativa
interna. A continuación, se describe técnicamente la lógica e implementación de la solución.

### Descripción de archivos y directorios

- `main.py`: Archivo principal de la API. Este es el archivo de entrada de la aplicación, donde se crea y configura la
  instancia de FastAPI. Aquí es donde se agregan rutas, middleware, y cualquier configuración general de la aplicación
  que necesita ser inicializada al arrancar.
- `depedencies.py`: Estas dependencias son funciones de utilidad, configuraciones de base de datos,
  servicios de dependencias (como clientes de API externos, configuración de caché, etc.), y cualquier otro objeto que
  necesite ser inyectado en los controladores/rutas de la aplicación.
- `routers/`: Este directorio contiene los diferentes enrutadores (routers) de la aplicación, organizados por
  funcionalidad o dominio. Cada archivo dentro de este directorio define rutas relacionadas con una parte específica de
  la aplicación. Para este caso en concreto:
    - `chat.py`: Contiene la funcionalidad y lógica del módulo del chat.
    - `rag_utils`: Contiene la funcionalidad, lógica y funciones auxiliares para el RAG.
    - `tender_agent`: Contiene la lógica del Agente de Licitaciones.
    - `evaluation`: Contiene la lógica de generación de la evaluación del modelo.
- `internals`: Este directorio alberga el código que es interno para la aplicación y no debe ser
  accesible directamente a través de las rutas de API públicas. Puede incluir lógica de administración, tareas de
  backend que se ejecutan en el servidor, scripts de mantenimiento, y otras utilidades internas.

## Guía de Uso

Lo primero a tener en cuenta es la variable de entorno para usar la API key de Azure OpenAI. Para ello, se debe crear un
archivo
`.env` en la raíz del proyecto con la siguiente estructura:

```shell
AZURE_OPENAI_API_KEY=XXXXXXXXXXXXXX # Sustituir por la API key de Azure OpenAI
AZURE_OPENAI_ENDPOINT=XXXXXXXXXXXXXXX
LANGCHAIN_API_KEY=XXXXXXXXXXXXXXXX
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_PROJECT="Your project name"
```

### Despliegue local :computer:

Para el despliegue en local, se debe instalar las dependencias del proyecto. Para ello, se puede utilizar el siguiente
comando:

```shell
pip install -r requirements.txt
```

Una vez instaladas correctamente todas las dependencias, se puede ejecutar la aplicación con el siguiente comando:

```shell
fastapi run main.py
```

### Despliegue con Docker :whale:

Para el despliegue en un contenedor docker, se debe construir la imagen del contenedor:

```shell
docker build -t api:latest .
```

Una vez construida la imagen, podemos ejecutarla del siguiente modo:
```shell
docker run -p 8000:8000 api 
```

### Documentación

Una vez desplegada la solución, es posible acceder a la misma a través de la URL base, obteniendo la siguiente respuesta:

```json
{"Status":"API is running","Time":"YYYY-MM-DD HH:MM:SS"}
```