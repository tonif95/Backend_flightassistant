from langchain.tools import tool
import smtplib
from email.mime.text import MIMEText
import os
import httpx

@tool
async def ryanair_flight_search(origen: str, destino: str, fecha: str, moneda: str = "EUR"):
    """
    Busca los vuelos más baratos en Ryanair de forma asíncrona.
    Args:
        origen: Código IATA del aeropuerto de origen (ej: MAD, BCN).
        destino: Código IATA del aeropuerto de destino (ej: LON, PAR).
        fecha: Fecha del vuelo en formato YYYY-MM-DD.
        moneda: Código de la moneda (default EUR).
    """

    url = "https://ryanair-api-hx0t.onrender.com/api/search-fares"
    params = {
        "from": origen,
        "to": destino,
        "date": fecha,
        "currency": moneda
    }

    # Usamos un cliente asíncrono
    # Aumentamos el timeout a 120 segundos porque la API de Render es lenta al despertar
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            # await es la clave: libera al servidor mientras espera
            response = await client.get(url, params=params)
            
            # Manejo específico del error 429 (Too Many Requests)
            if response.status_code == 429:
                return "⚠️ La API de vuelos está saturada momentáneamente. Por favor, espera 1 minuto e inténtalo de nuevo."

            # Si es un error 502 (Bad Gateway), suele ser porque se está despertando
            if response.status_code == 502:
                return "⚠️ El servidor de vuelos se está reiniciando (Cold Start). Por favor, intenta la misma búsqueda en 30 segundos."

            response.raise_for_status()
            return response.json()
        
        except httpx.HTTPStatusError as e:
            return {"error": f"Error HTTP {e.response.status_code}: {str(e)}"}
        except httpx.RequestError as e:
            return {"error": f"Error de conexión: {str(e)}"}
        except Exception as e:
            return {"error": f"Error inesperado: {str(e)}"}
    
@tool
def send_email(subject: str, body: str, destinatario: str):
    """
    Envía un correo electrónico con la información proporcionada.
    Útil para enviar resúmenes de vuelos al usuario.
    """
    # --- CONFIGURACIÓN DE GMAIL ---
    REMITENTE = os.getenv("GMAIL_SENDER_EMAIL")  # Tu dirección de Gmail
    # ¡IMPORTANTE! Usa la Contraseña de Aplicación de 16 dígitos
    PASSWORD = os.getenv("GMAIL_APP_PASSWORD") 
    DESTINATARIO = destinatario 
    if not REMITENTE or not PASSWORD:
        return "❌ Error de configuración: Credenciales de email no encontradas en el entorno."
    # --- DATOS DEL MENSAJE ---
    ASUNTO = subject
    CUERPO = body

    # 1. Crear el objeto del mensaje
    msg = MIMEText(CUERPO)
    msg['Subject'] = ASUNTO
    msg['From'] = REMITENTE
    msg['To'] = DESTINATARIO

    # 2. Establecer la conexión y enviar
    servidor = None
    try:
        # Servidor y puerto SMTP de Gmail
        servidor = smtplib.SMTP('smtp.gmail.com', 587)
        
        # Iniciar la encriptación TLS (es crucial para Gmail)
        servidor.starttls() 
        
        # Autenticación con tu correo y la Contraseña de Aplicación
        servidor.login(REMITENTE, PASSWORD)
        
        # Enviar el correo
        servidor.sendmail(REMITENTE, DESTINATARIO, msg.as_string())
        
        print("✅ Correo enviado exitosamente usando Gmail y Python.")

    except Exception as e:
        print(f"❌ Error al enviar el correo: {e}")

    finally:
        # Cerrar la conexión
        if 'servidor':
            servidor.quit()