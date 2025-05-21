# func/vr_utils.py
import shutil, os

# constantes — ajuste aqui se algum dia mudar de pasta
DEFAULT_SRC_PATH = "/media/oculos/SessionData/resultado.json"
DEFAULT_DST_DIR  = "../dados_sessoes"

def extrair_json(nome_limpo: str,
                 src_path: str = DEFAULT_SRC_PATH,
                 dst_dir:  str = DEFAULT_DST_DIR) -> str:
    """
    Copia o JSON do VR (src_path) para dst_dir, nomeando como {nome_limpo}_VR.json.
    Retorna o path completo do arquivo salvo.
    """
    os.makedirs(dst_dir, exist_ok=True)
    filename = f"{nome_limpo}_VR.json"
    dst_path = os.path.join(dst_dir, filename)
    shutil.copy(src_path, dst_path)
    return dst_path
