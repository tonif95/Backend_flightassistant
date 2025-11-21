import requests
from langchain.tools import tool
import smtplib
from email.mime.text import MIMEText
import os

@tool
def ryanair_flight_search(origen: str, destino: str, fecha: str, moneda: str = "EUR"):
    """
    Busca los vuelos más baratos en Ryanair para una fecha específica.
    ...
    """

    url = "https://ryanair-api-hx0t.onrender.com/api/search-fares"

    params = {
        "from": origen,
        "to": destino,
        "date": fecha,
        "currency": moneda
    }

    try:
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e: 
        return {"error": f"Error de conexión: {str(e)}"}
    
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