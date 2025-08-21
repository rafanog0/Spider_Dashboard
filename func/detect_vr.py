# func/vr_utils.py
import pyopenxr as xr   # ✅ troca do import

# 1) Cria a instância **uma só vez** no módulo
_app_info = xr.XrApplicationInfo(
    applicationName="StreamlitVRDetector".encode("utf-8"),
    applicationVersion=1,
    engineName="None".encode("utf-8"),
    engineVersion=1,
    apiVersion=xr.XR_CURRENT_API_VERSION
)
_create_info = xr.XrInstanceCreateInfo(
    next=None,
    createFlags=0,
    applicationInfo=_app_info,
    enabledApiLayerCount=0,
    enabledApiLayerNames=None,
    enabledExtensionCount=0,
    enabledExtensionNames=None
)
_instance = xr.xrCreateInstance(_create_info)

def detectar_vr() -> bool:
    """
    Retorna True se um HMD OpenXR estiver disponível no sistema.
    Não faz nenhuma espera — apenas checa e retorna imediatamente.
    """
    res, _ = xr.xrGetSystem(
        _instance,
        formFactor=xr.XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
    )
    return (res == xr.XR_SUCCESS)