# API-Asistente

API

### Descripción de archivos y directorios

- `main.py`: Archivo principal de la API. Este es el archivo de entrada de la aplicación, donde se crea y configura la
  instancia de FastAPI. Aquí es donde se agregan rutas, middleware, y cualquier configuración general de la aplicación
  que necesita ser inicializada al arrancar.
- `depedencies.py`: Estas dependencias son funciones de utilidad, configuraciones de base de datos,
  servicios de dependencias (como clientes de API externos, configuración de caché, etc.), y cualquier otro objeto que
  necesite ser inyectado en los controladores/rutas de la aplicación.
- `routers/`: Este directorio contiene los diferentes enrutadores (routers) de la aplicación, organizados por
  funcionalidad o dominio. Cada archivo dentro de este directorio define rutas relacionadas con una parte específica de
  la aplicación.
- `internals`: Este directorio alberga el código que es interno para la aplicación y no debe ser
  accesible directamente a través de las rutas de API públicas. Puede incluir lógica de administración, tareas de
  backend que se ejecutan en el servidor, scripts de mantenimiento, y otras utilidades internas.