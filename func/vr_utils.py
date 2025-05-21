import time
import shutil
import os
import json
import sqlite3
from datetime import datetime
import xr

# Constantes para paths padrão
DEFAULT_SRC_PATH = "/media/oculos/SessionData/resultado.json"
DEFAULT_DST_DIR  = "../dados_sessoes"

# Instância OpenXR criada uma só vez
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


def detectar_vr(poll_interval: float = 0.5) -> bool:
    """
    Bloqueia até que um HMD OpenXR seja detectado.
    Retorna True assim que conectado.
    """
    while True:
        res, _ = xr.xrGetSystem(
            _instance,
            formFactor=xr.XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY
        )
        if res == xr.XR_SUCCESS:
            return True
        time.sleep(poll_interval)


def extrair_json(nome_limpo: str,
                  src_path: str = DEFAULT_SRC_PATH,
                  dst_dir: str = DEFAULT_DST_DIR) -> str:
    """
    Copia o JSON do VR (src_path) para dst_dir,
    nomeando como {nome_limpo}_VR.json. Retorna o path destino.
    """
    os.makedirs(dst_dir, exist_ok=True)
    filename = f"{nome_limpo}_VR.json"
    dst_path = os.path.join(dst_dir, filename)
    shutil.copy(src_path, dst_path)
    return dst_path


def process_and_save(db_path: str,
                     dashboard_json: str,
                     vr_json: str,
                     participante_id: int):
    """
    Processa os JSONs gerados pela dashboard e pelo VR e insere nos
    respectivos esquemas fase1, fase2 e fase3.

    - dashboard_json: path para JSON com 'bpm_dados' e 'timestamps' ("HH:MM:SS").
    - vr_json: path para JSON com campos de fase, cada um contendo
      'start', 'end' no formato "HH:MM:SS" e demais atributos 'tempo...'.
    - participante_id: chave estrangeira no banco.

    Regras:
      * Para cada fase, filtrar BPMs cujos timestamps estejam entre start e end.
      * session_bpm armazenado como string array JSON.
      * Atributos 'tempo...' armazenar valor inteiro antes do ponto e duas casas decimais.
    """
    # carrega JSONs
    with open(dashboard_json, encoding='utf-8') as f:
        dash = json.load(f)
    with open(vr_json, encoding='utf-8') as f:
        vr = json.load(f)

    # conecta e prepara cursor
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # função auxiliar para parse HH:MM:SS
    def parse_hhmmss(ts: str) -> datetime.time:
        return datetime.strptime(ts, "%H:%M:%S").time()

    # pre-processa dashboard timestamps em tempos
    dash_times = [parse_hhmmss(t) for t in dash['timestamps']]
    dash_bpms  = dash['bpm_dados']

    # insere cada fase
    for fase_key, table in [('fase1', 'fase1'), ('fase2', 'fase2'), ('fase3', 'fase3')]:
        if fase_key not in vr:
            continue
        fase = vr[fase_key]
        # obtém intervalos
        start = parse_hhmmss(fase['start'])
        end   = parse_hhmmss(fase['end'])
        # filtra bpms
        bpms_fase = [b for b, t in zip(dash_bpms, dash_times) if start <= t <= end]
        session_bpm_str = json.dumps(bpms_fase)

        # prepara valores de tempo arredondados
        tempo_kwargs = {}
        for k, v in fase.items():
            if k.startswith('tempo') and isinstance(v, (int, float)):
                tempo_kwargs[k] = f"{v:.2f}"

        # monta insert conforme tabela
        if table == 'fase1':
            cur.execute(
                "INSERT INTO fase1(participante_id, session_bpm, tempo_finalizacao, tempo_encontrar_aranha)"
                " VALUES(?,?,?,?)",
                (participante_id,
                 session_bpm_str,
                 tempo_kwargs.get('tempoFinalizacao','0.00'),
                 tempo_kwargs.get('tempoAranha','0.00'))
            )
        elif table == 'fase2':
            cur.execute(
                "INSERT INTO fase2(participante_id, session_bpm, tempo_finalizacao, tempo_encontrar_copo)"
                " VALUES(?,?,?,?)",
                (participante_id,
                 session_bpm_str,
                 tempo_kwargs.get('tempoFinalizacao','0.00'),
                 tempo_kwargs.get('tempoCopo','0.00'))
            )
        else:  # fase3
            # conta quantas chaves timeline
            tempos = [v for k, v in tempo_kwargs.items() if k.startswith('tempoAranha')]
            total = len(tempos)
            # insere todos campos tempoAranha1..N
            columns = [f"tempo_encontrar_aranha_{i+1}" for i in range(total)]
            values  = [tempos[i] for i in range(total)]
            cols_sql = ",".join(["participante_id","session_bpm","tempo_finalizacao"] + columns + ["total_aranhas_entradas"])
            q_marks = ",".join(["?"] * (3 + total + 1))
            cur.execute(
                f"INSERT INTO fase3({cols_sql}) VALUES({q_marks})",
                (participante_id, session_bpm_str, tempo_kwargs.get('tempoFinalizacao','0.00'), *values, total)
            )
    conn.commit()
    conn.close()
