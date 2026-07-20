import os
import base64
from langchain_core.tools import tool
from backend.config import settings
from backend.logging_config import get_logger

log = get_logger(__name__)

@tool
def analyze_screen(prompt: str = "Describe what is currently visible on the screen.") -> str:
    """Take a screenshot of the user's screen and analyze/explain it.

    Use this when the user asks 'what is on my screen', 'read this error on my screen',
    'explain this image/window', or wants you to look at something currently open.

    Args:
        prompt: Specific question or instruction for analyzing the screen (e.g. 'what app is open?', 'read the code visible in the editor').
    """
    log.info("[screen_analyzer] Capture and analysis requested with prompt: %r", prompt)
    
    # 1. Check/Install Pillow dependency
    try:
        from PIL import ImageGrab
    except ImportError:
        return (
            "Error: The 'Pillow' library is required to capture screenshots. "
            "Please run: pip install Pillow"
        )

    # 2. Take screenshot
    try:
        # Create output directory if it doesn't exist
        os.makedirs(".ai_os", exist_ok=True)
        screenshot_path = os.path.join(".ai_os", "screenshot.png")
        
        # Grab screen
        screenshot = ImageGrab.grab()
        screenshot.save(screenshot_path, "PNG")
        log.info("[screen_analyzer] Screenshot saved to %s", screenshot_path)
    except Exception as exc:
        log.exception("[screen_analyzer] Failed to capture screenshot")
        return f"Failed to capture screenshot: {exc}"

    # 3. Base64 encode the image
    try:
        with open(screenshot_path, "rb") as img_file:
            base64_image = base64.b64encode(img_file.read()).decode("utf-8")
    except Exception as exc:
        log.exception("[screen_analyzer] Failed to read screenshot file")
        return f"Failed to read screenshot file: {exc}"

    # 4. Invoke LLM Vision model
    # Try Gemini first, then fall back to Groq Vision
    errors = []
    
    # Try Gemini fallback
    if settings.GEMINI_API_KEY:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            from langchain_core.messages import HumanMessage
            
            log.info("[screen_analyzer] Analyzing using Gemini (gemini-2.0-flash)")
            model = ChatGoogleGenerativeAI(
                model=settings.GEMINI_MODEL,
                google_api_key=settings.GEMINI_API_KEY,
                temperature=settings.GEMINI_TEMPERATURE,
                max_retries=0,
            )
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": f"You are the Vision module of the AI OS. User request: {prompt}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            )
            
            response = model.invoke([message])
            return f"[Gemini Vision Response]\n{response.content}"
        except Exception as exc:
            log.warning("[screen_analyzer] Gemini vision failed: %s", exc)
            errors.append(f"Gemini error: {exc}")

    # Try Groq fallback
    groq_keys = settings.groq_api_keys
    if groq_keys:
        for idx, key in enumerate(groq_keys, start=1):
            try:
                from langchain_groq import ChatGroq
                from langchain_core.messages import HumanMessage
                
                # Determine which model to use. Llama-3.2 vision is best for this on Groq.
                model_name = os.getenv("GROQ_VISION_MODEL") or "llama-3.2-11b-vision-preview"
                log.info("[screen_analyzer] Analyzing using Groq (model=%s, key=%d)", model_name, idx)
                
                model = ChatGroq(
                    model=model_name,
                    api_key=key,
                    temperature=settings.GROQ_TEMPERATURE,
                    max_retries=0,
                )
                
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": f"You are the Vision module of the AI OS. User request: {prompt}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        },
                    ]
                )
                
                response = model.invoke([message])
                return f"[Groq Vision Response ({model_name})]\n{response.content}"
            except Exception as exc:
                log.warning("[screen_analyzer] Groq vision failed with key %d: %s", idx, exc)
                errors.append(f"Groq key {idx} error: {exc}")

    # If no keys or all failed
    if not settings.GEMINI_API_KEY and not groq_keys:
        return (
            "Error: No LLM keys available for vision analysis. "
            "Please configure GEMINI_API_KEY or GROQ_API_KEY_1 in your .env file."
        )
        
    return f"Vision analysis failed.\nDetails:\n" + "\n".join(errors)
