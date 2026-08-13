from jinja2 import Template
from datetime import datetime

HTML_TEMPLATE = """
<html>
<head><meta charset="utf-8"><title>Reporte EDA + IA</title>
<style>
  body { font-family: Arial, sans-serif; margin: 40px; }
  h1 { color: #2c3e50; }
  .insight { padding: 8px; margin: 6px 0; border-left: 4px solid #3498db; background: #f4f8fb; }
  .high { border-left-color: #e74c3c; }
  .medium { border-left-color: #f39c12; }
</style>
</head>
<body>
  <h1>Reporte de Análisis Exploratorio</h1>
  <p>Generado: {{ fecha }}</p>
  <h2>Hallazgos</h2>
  {% for i in insights %}
    <div class="insight {{ i.severity }}">{{ i.text }}</div>
  {% endfor %}
</body>
</html>
"""

def export_html(insights: list[dict], output_path: str = "reporte.html"):
    template = Template(HTML_TEMPLATE)
    html = template.render(insights=insights,
                            fecha=datetime.now().strftime("%Y-%m-%d %H:%M"))
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
