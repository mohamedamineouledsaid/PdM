import sys
sys.path.insert(0, '.')
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Fixed helper function output simulation
def fixed_radial_confidence_gauge_svg(confidence_pct=95.4, size=110):
    import numpy as np
    radius = 42
    circumference = 2 * np.pi * radius
    stroke_dasharray = circumference
    stroke_dashoffset = circumference - (confidence_pct / 100.0) * circumference
    return f'''<div style="text-align: center;"><svg width="{size}" height="{size}" viewBox="0 0 100 100"><circle cx="50" cy="50" r="{radius}" fill="none" stroke="#1e293b" stroke-width="9" /><circle cx="50" cy="50" r="{radius}" fill="none" stroke="url(#cyanLimeGradient)" stroke-width="9" stroke-dasharray="{stroke_dasharray:.1f}" stroke-dashoffset="{stroke_dashoffset:.1f}" stroke-linecap="round" transform="rotate(-90 50 50)" /><defs><linearGradient id="cyanLimeGradient" x1="0%" y1="0%" x2="100%" y2="100%"><stop offset="0%" stop-color="#00e5ff" /><stop offset="100%" stop-color="#00ff66" /></linearGradient></defs><text x="50" y="47" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="16" font-weight="800" fill="#00ff66">{confidence_pct:.1f}%</text><text x="50" y="62" text-anchor="middle" font-family="'JetBrains Mono', monospace" font-size="7" font-weight="700" fill="#64748b">HIGH CONF</text></svg></div>'''

radial_gauge_html = fixed_radial_confidence_gauge_svg(confidence_pct=95.4, size=105)

selected_horizon = "6h"
traj_color = "#00ff66"
predicted_dg = 1234.5
current_dg = 1200.0
abs_diff = 34.5
pct_diff = 2.87
scada_t0_str = "2023-12-31 21:00:00"
scada_tf_str = "2024-01-01 03:00:00"
traj_icon = "▲"
traj_status = "INCREASE / RECOVERY"

html_str = f"""<div class="hero-forecast-panel" style="border-top: 4px solid {traj_color};">
    <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid rgba(255, 255, 255, 0.08); padding-bottom: 12px; margin-bottom: 16px; flex-wrap: wrap; gap: 10px;">
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.05rem; font-weight: 800; color: var(--scada-text-bright); display: flex; align-items: center; gap: 10px;">
            <span>DEEP LEARNING FORECASTING ENGINE</span>
            <span class="status-badge" style="background: rgba(255,255,255,0.06); color: #cbd5e1; border: 1px solid #1f293d;">MODEL: CNN-TRANSFORMER V1</span>
            <span class="status-badge" style="background: rgba(0, 229, 255, 0.1); color: #00e5ff; border: 1px solid #00e5ff;">HORIZON: {selected_horizon.upper()}</span>
        </div>
        <div class="status-badge" style="background: rgba(0, 255, 102, 0.12); color: {traj_color}; border: 1px solid {traj_color}; font-size: 0.85rem;">
            {traj_icon} TRAJECTORY: {traj_status}
        </div>
    </div>
    <div style="display: grid; grid-template-columns: 2.2fr 1fr 1fr 1fr 1fr 1fr; gap: 14px; align-items: center;">
        <div class="hero-big-card" style="border-color: {traj_color};">
            <div style="font-size: 0.75rem; font-weight: 700; color: var(--scada-text-muted); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">PREDICTED GAS FLOW (+{selected_horizon.upper()})</div>
            <div class="hero-big-val" style="color: {traj_color};">{predicted_dg:,.2f} <span style="font-size: 1rem; color: var(--scada-text-muted);">M3/h</span></div>
        </div>
        <div style="background: rgba(9, 13, 22, 0.6); padding: 12px; border-radius: 6px; border: 1px solid var(--scada-card-border);">
            <div style="font-size: 0.7rem; color: var(--scada-text-muted); font-weight: 700; text-transform: uppercase;">Current DG (t0)</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 800; color: var(--scada-text-bright); margin-top: 2px;">{current_dg:,.2f} <span style="font-size: 0.7rem; color: var(--scada-text-muted);">M3/h</span></div>
        </div>
        <div style="background: rgba(9, 13, 22, 0.6); padding: 12px; border-radius: 6px; border: 1px solid var(--scada-card-border);">
            <div style="font-size: 0.7rem; color: var(--scada-text-muted); font-weight: 700; text-transform: uppercase;">Expected Δ Change</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 800; color: {traj_color}; margin-top: 2px;">{abs_diff:+,.2f} <span style="font-size: 0.7rem; color: var(--scada-text-muted);">M3/h</span></div>
        </div>
        <div style="background: rgba(9, 13, 22, 0.6); padding: 12px; border-radius: 6px; border: 1px solid var(--scada-card-border);">
            <div style="font-size: 0.7rem; color: var(--scada-text-muted); font-weight: 700; text-transform: uppercase;">Percentage Change</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 800; color: {traj_color}; margin-top: 2px;">{pct_diff:+,.2f}%</div>
        </div>
        <div style="background: rgba(9, 13, 22, 0.6); padding: 12px; border-radius: 6px; border: 1px solid var(--scada-card-border);">
            <div style="font-size: 0.7rem; color: var(--scada-text-muted); font-weight: 700; text-transform: uppercase;">Input Window</div>
            <div style="font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 800; color: var(--scada-text-bright); margin-top: 2px;">360 Min</div>
        </div>
        <div>{radial_gauge_html}</div>
    </div>
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px;">
        <div style="background: rgba(9, 13, 22, 0.5); padding: 8px 12px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.05); font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
            <span style="color: var(--scada-text-muted);">Prediction Reference (t0):</span> <strong style="color: #00e5ff;">{scada_t0_str}</strong>
        </div>
        <div style="background: rgba(9, 13, 22, 0.5); padding: 8px 12px; border-radius: 5px; border: 1px solid rgba(255,255,255,0.05); font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">
            <span style="color: var(--scada-text-muted);">Target Forecast (+{selected_horizon}):</span> <strong style="color: {traj_color};">{scada_tf_str}</strong>
        </div>
    </div>
</div>"""

print("=== FIXED RAW HTML OUTPUT ===")
lines = html_str.split('\n')
for idx, line in enumerate(lines):
    print(f"{idx+1:3d}: {repr(line)}")
