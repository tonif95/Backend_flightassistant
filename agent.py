from dotenv import load_dotenv
# Manejo de compatibilidad de versiones de Python para Tipado
try:
    from typing import TypedDict
except Exception:
    from typing_extensions import TypedDict

# Importaciones de LangGraph y LangChain
from langgraph.graph.message import add_messages
from typing import List, Any, Dict, Annotated, Optional, Literal
import uuid
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from datetime import datetime
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
import asyncio
from pydantic import BaseModel, Field

# --- IMPORTACIÓN DE HERRAMIENTAS ---

try:
    from flightassistant_tools import ryanair_flight_search, send_email
except ImportError:
    pass

# Cargamos las variables de entorno (.env)
load_dotenv(override=True)

# --- DEFINICIÓN DEL ESTADO (STATE) ---
# El estado define la estructura de datos que se pasa entre los nodos del grafo.
class State(TypedDict):
    # 'add_messages' asegura que los nuevos mensajes se añadan a la lista existente (append) en lugar de sobrescribirla
    messages: Annotated[List[BaseMessage], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool

# --- ESTRUCTURA DE SALIDA DEL EVALUADOR ---
# Usamos Pydantic para forzar al LLM evaluador a responder con un formato estructurado JSON
class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback sobre la respuesta del asistente")
    success_criteria_met: bool = Field(description="True si se han cumplido los criterios de éxito")
    user_input_needed: bool = Field(description="True si se necesita más información del usuario o el asistente está atascado")

# --- CLASE PRINCIPAL DEL ASISTENTE ---
class FlightAssistant:
    def __init__(self, memory):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = [ryanair_flight_search, send_email]
        self.memory = memory # Checkpointer para persistencia (SQLite)
        self.graph = None
        

    async def setup(self):
        """Inicializa los modelos y construye el grafo."""
        
        # Configurar el modelo del Worker (el que hace el trabajo)
        worker_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)

        # Configurar el modelo del Evaluador (el que critica el trabajo)
        evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)
        
        # Construir el grafo
        await self.build_graph()

    # --- NODOS DEL GRAFO ---

    def worker(self, state: State) -> Dict[str, Any]:
        """
        Nodo Worker: Genera respuestas o llamadas a herramientas.
        """
        # Instrucciones del sistema dinámicas
        system_message_content = f"""You are a helpful assistant equipped with web tools and FLIGHT search tools, and EMAIL tools.
            You keep working on a task until either you have a question or clarification for the user, or the success criteria is met.
            The current date is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.
            
            This is the success criteria:
            {state.get("success_criteria", "Provide a helpful answer")}

            ### CRITICAL ERROR HANDLING (HIGHEST PRIORITY):
            If the 'ryanair_flight_search' tool returns a message mentioning "Cold Start", "reiniciando", or "502":
            1. DO NOT CALL THE TOOL AGAIN immediately.
            2. STOP and inform the user: "⚠️ El servidor de vuelos se está iniciando (Cold Start). Por favor, espera 30 segundos y vuelve a preguntarme."
            3. Do not try to fix it yourself, just report it and stop.

            ### RULES FOR FLIGHT SEARCH:
            1. If the user asks for flights, use the 'ryanair_flight_search' tool.
            2. You MUST convert city names to IATA codes yourself (e.g., Madrid -> MAD).
            3. Format dates strictly as YYYY-MM-DD.
            
            ### RULES FOR EMAIL:
            1. OFFER EMAIL: After presenting the results, YOU MUST ASK the user: "Do you want me to email you this summary?".
            2. SEND EMAIL: 
               - If the user says YES: Ask for their email address (if you don't know it yet).
               - Once you have the email, use 'send_email' tool to send the summary.
               - Subject should be descriptive (e.g., "Flight Summary: MAD to LON").
        """
        
        # Si hubo feedback negativo anterior, lo añadimos al prompt
        if state.get("feedback_on_work"):
            system_message_content += f"""
            Previously your reply was rejected. Feedback: {state["feedback_on_work"]}
            Please fix this.
            """
        
        # Gestión de mensajes: Insertamos el SystemMessage al principio
        messages = state["messages"]
        # Filtramos mensajes de sistema antiguos para no acumularlos si el bucle se repite mucho
        filtered_messages = [m for m in messages if not isinstance(m, SystemMessage)]
        final_messages = [SystemMessage(content=system_message_content)] + filtered_messages
        
        response = self.worker_llm_with_tools.invoke(final_messages)
        
        # Devolvemos el nuevo mensaje para que se añada al estado
        return {"messages": [response]}
    
    def format_conversation(self, messages: List[BaseMessage]) -> str:
        """Helper para formatear el chat como texto plano para el evaluador."""
        conversation = "Conversation history:\n\n"
        for message in messages:
            if isinstance(message, HumanMessage):
                conversation += f"User: {message.content}\n"
            elif isinstance(message, AIMessage):
                # Manejo seguro si content es None (p.ej. solo tool_call)
                text = message.content if message.content else "[Tool Call]"
                conversation += f"Assistant: {text}\n"
            elif isinstance(message, SystemMessage):
                continue # Omitimos system messages para el evaluador para no confundirlo
        return conversation
    
    def evaluator(self, state: State) -> Dict[str, Any]:
        """
        Nodo Evaluador: Juzga la última respuesta del Worker.
        """
        # Obtenemos el último mensaje generado por el worker
        last_message = state["messages"][-1]
        last_response = last_message.content if last_message.content else "[Action/Tool Call]"

        system_prompt = """You are an evaluator. Assess if the Assistant met the success criteria."""
        
        user_prompt = f"""
        Conversation:
        {self.format_conversation(state["messages"])}

        Success Criteria:
        {state.get("success_criteria")}

        Last Assistant Response:
        {last_response}

        Did the assistant meet the criteria? Does it need more user input?
        """

        # Invocamos al LLM con salida estructurada
        eval_result = self.evaluator_llm_with_output.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])

        # Devolvemos las actualizaciones del estado
        # Nota: Añadimos el feedback como un mensaje invisible para el usuario pero visible en el historial
        # o simplemente actualizamos las variables de control.
        return {
            "feedback_on_work": eval_result.feedback,
            "success_criteria_met": eval_result.success_criteria_met,
            "user_input_needed": eval_result.user_input_needed,
        }

    # --- FUNCIONES DE ENRUTAMIENTO (EDGES) ---

    def worker_router(self, state: State) -> Literal["tools", "evaluator"]:
        """Decide si el worker quiere usar una herramienta o si ha terminado de hablar."""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return "evaluator"
    
    def route_based_on_evaluation(self, state: State) -> Literal["END", "worker"]:
        """Decide si terminamos el flujo o volvemos al worker para corregir."""
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END" # Mapeado a END en el grafo
        else:
            return "worker" # Volver a intentar

    # --- CONSTRUCCIÓN DEL GRAFO ---

    async def build_graph(self):
        graph_builder = StateGraph(State)
        
        # Añadir Nodos
        graph_builder.add_node("worker", self.worker)
        graph_builder.add_node("tools", ToolNode(tools=self.tools))
        graph_builder.add_node("evaluator", self.evaluator)
        
        # Añadir Aristas (Flujo)
        graph_builder.add_edge(START, "worker")
        
        # Salida del Worker: ¿Herramientas o Evaluación?
        graph_builder.add_conditional_edges(
            "worker", 
            self.worker_router, 
            {"tools": "tools", "evaluator": "evaluator"}
        )
        
        # Salida de Herramientas: Siempre vuelve al worker
        graph_builder.add_edge("tools", "worker")
        
        # Salida del Evaluador: ¿Fin o Reintentar?
        # AQUÍ ESTABA EL ERROR PRINCIPAL ANTES: Debe usar route_based_on_evaluation
        graph_builder.add_conditional_edges(
            "evaluator", 
            self.route_based_on_evaluation, 
            {"worker": "worker", "END": END}
        )
        
        # Compilación con memoria
        self.graph = graph_builder.compile(checkpointer=self.memory)
    
    async def run_superstep(self, user_input: str, thread_id: str):
        """
        Ejecuta un paso.
        """
        config = {"configurable": {"thread_id": thread_id}} # Usamos el ID que viene de la API

        initial_input = {
            "messages": [HumanMessage(content=user_input)],
            "success_criteria": "Answer clearly. Offer email if relevant.",
            "success_criteria_met": False,
            "user_input_needed": False,
            "feedback_on_work": None
        }

        # Invocamos el grafo
        # Usamos invoke o ainvoke. Esto ejecutará el bucle worker -> evaluator -> worker hasta que termine.
        final_state = await self.graph.ainvoke(initial_input, config=config)
        
        # Extraemos la última respuesta del asistente del estado final
        # Buscamos el último mensaje que sea de tipo AIMessage y que tenga contenido (no solo tool calls)
        messages = final_state["messages"]
        last_msg = messages[-1]
        
        return last_msg.content

