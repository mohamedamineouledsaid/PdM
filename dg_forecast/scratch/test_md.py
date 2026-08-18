import markdown

s = """<div class="hero">
    <div>
        
    <div style="text-align: center;">
        <svg></svg>
    </div>
    
            </div>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 14px;">
            <div>test</div>
        </div>
    </div>"""

print("--- RESULT WITH BLANK LINE & 12 SPACES ---")
print(markdown.markdown(s))
