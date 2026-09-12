import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import os
import time
import re

queries = [
    "Ignore Previous Prompt Attack Techniques for Language Models",
    "AgentDojo A Dynamic Environment to Evaluate Attacks",
    "ToolEmu Identifying the Risks of LM Agents",
    "InjecAgent Benchmarking Indirect Prompt Injections",
    "CaMeL Defeating Prompt Injections by Design",
    "AgentSight System-Level Observability for AI Agents",
    "ActPlane Programmable OS-Level Policy Enforcement",
    "A Survey on the Safety and Security Threats of Computer-Using Agents",
    "Toward Secure LLM Agents Threat Surfaces Attacks Defenses",
    "Open Challenges in Multi-Agent Security",
    "Prompt Injection Attacks in LLMs and AI Agent Systems A Comprehensive Review"
]

output_dir = r"c:\Users\rvsre\.gemini\antigravity\scratch\os_level_guardrails\papers"

for q in queries:
    try:
        # We'll use title search for better accuracy
        search_query = f'ti:"{q.split(":")[0][:40]}"' 
        # Fallback to ALL search
        search_query_all = urllib.parse.quote(f"all:{q}")
        url = f"http://export.arxiv.org/api/query?search_query={search_query_all}&start=0&max_results=1"
        
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req).read()
        root = ET.fromstring(response)
        
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        entry = root.find('atom:entry', ns)
        
        if entry is not None:
            title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
            pdf_url = None
            for link in entry.findall('atom:link', ns):
                if link.attrib.get('title') == 'pdf':
                    pdf_url = link.attrib['href']
                    # Use standard arxiv pdf url format
                    pdf_url = pdf_url.replace("http://arxiv.org/abs/", "https://arxiv.org/pdf/") + ".pdf"
                    if not pdf_url.endswith(".pdf.pdf"):
                        pass
                    break
            
            if not pdf_url:
                # Fallback to id
                id_url = entry.find('atom:id', ns).text
                if 'arxiv.org/abs/' in id_url:
                    arxiv_id = id_url.split('arxiv.org/abs/')[-1]
                    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"

            if pdf_url:
                safe_title = re.sub(r'[^a-zA-Z0-9 ]', '', title)[:80].strip()
                pdf_path = os.path.join(output_dir, f"{safe_title}.pdf")
                print(f"Downloading: {title}\n -> {pdf_url}")
                urllib.request.urlretrieve(pdf_url, pdf_path)
            else:
                print(f"No PDF link found for: {title}")
        else:
            print(f"Not found on ArXiv: {q}")
    except Exception as e:
        print(f"Error searching for '{q}': {e}")
    time.sleep(3)
