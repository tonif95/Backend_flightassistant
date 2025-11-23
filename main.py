import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Importamos tu clase del archivo agent.py
from agent import FlightAssistant

# --- 1. GESTIÓN DEL CICLO DE VIDA (LIFESPAN) ---
# Esto se ejecuta una sola vez al arrancar el servidor.
# Es donde conectamos la base de datos y preparamos el agente.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Iniciando servidor y conectando a memoria...")
    
    # NOTA PARA RENDER: 
    # En la versión gratuita/web services de Render, el sistema de archivos es efímero.
    # 'memory.db' se borrará cada vez que redespliegues. 
    # Para producción real, en el futuro cambiaremos esto por Postgres.
    async with AsyncSqliteSaver.from_conn_string("memory.db") as checkpointer:
        # Inicializamos el agente
        assistant = FlightAssistant(memory=checkpointer)
        await assistant.setup()
        
        # Guardamos la instancia del agente en la app para usarla en los endpoints
        app.state.agent = assistant
        
        print("✅ Agente listo y esperando peticiones.")
        yield
        
    print("🛑 Apagando servidor y cerrando conexiones...")

# --- 2. CONFIGURACIÓN DE LA APP ---
app = FastAPI(
    title="Flight Assistant API", 
    version="1.0.0",
    lifespan=lifespan
)

# --- 3. CONFIGURACIÓN DE CORS (CRUCIAL PARA REACT) ---
# Permite que tu frontend (que estará en otro dominio) hable con este backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-flightassistant.onrender.com/"],  # En producción, cambia "*" por la URL de tu frontend en Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. MODELOS DE DATOS (Pydantic) ---
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_user" # Identificador único de usuario/sesión

class ChatResponse(BaseModel):
    response: str
    status: str = "success"

# --- 5. ENDPOINTS ---

@app.get("/")
async def health_check():
    """Endpoint para verificar que el servidor está vivo (Health Check)."""
    return {"status": "online", "service": "Flight Assistant AI"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Recibe el mensaje del usuario y devuelve la respuesta del agente.
    """
    # Recuperamos el agente inicializado en lifespan
    agent: FlightAssistant = app.state.agent
    
    try:
        # Ejecutamos el agente pasando el mensaje y el ID de hilo
        respuesta_texto = await agent.run_superstep(
            user_input=request.message, 
            thread_id=request.thread_id
        )
        
        return ChatResponse(response=str(respuesta_texto))
    
    except Exception as e:
        print(f"❌ Error procesando solicitud: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 6. ARRANQUE DEL SERVIDOR (PARA RENDER) ---
if __name__ == "__main__":
    # Render inyecta la variable de entorno PORT. 
    # Si no existe (local), usa 8000.
    port = int(os.environ.get("PORT", 8000))
    
    # host="0.0.0.0" es OBLIGATORIO para despliegues en la nube (Render, Docker, etc)
    uvicorn.run(app, host="0.0.0.0", port=port)