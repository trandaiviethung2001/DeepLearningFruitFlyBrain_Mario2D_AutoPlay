"""Builds the User-Friendly, Semantically Color-Coded Decision & Firing Inspector Dashboard."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = REPO_ROOT / "data" / "telemetry_dashboard_data.json"
HTML_OUT = REPO_ROOT / "dashboards" / "malecns_decision_dashboard.html"
ART_OUT = Path("/Users/trandaiviethung/.gemini/antigravity/brain/c20d0836-81e3-4066-98b9-aa1ff9b7cca9/malecns_decision_dashboard.html")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>MaleCNS Decision & Neural Firing Inspector</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    :root {
      --bg-main: #0b0f19;
      --card-bg: #131c2e;
      --card-muted: #1c283f;
      --border-color: #26354f;
      --border-hover: #3b4f73;
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
      --text-dim: #64748b;
      --color-vision: #f59e0b;
      --color-core: #06b6d4;
      --color-motor: #f43f5e;
      --color-success: #10b981;
    }
    body {
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0b0f19; }
    ::-webkit-scrollbar-thumb { background: #26354f; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #3b4f73; }
  </style>
</head>
<body class="antialiased p-4 md:p-6 lg:p-8 min-h-screen">
  <div class="max-w-7xl mx-auto space-y-5">

    <!-- Header Section -->
    <header class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-2xl p-5 md:p-6 shadow-xl flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
      <div>
        <div class="flex items-center gap-2 mb-2">
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30">
            Drosophila MaleCNS Connectome
          </span>
          <span class="px-2.5 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/15 text-cyan-400 border border-cyan-500/30">
            World 1-1 Flagpole Win (x=3,161)
          </span>
        </div>
        <h1 class="text-2xl md:text-3xl font-extrabold tracking-tight text-white font-mono">
          MaleCNS Policy Decision & Firing Inspector
        </h1>
        <p class="text-sm text-[var(--text-muted)] mt-1">
          Inspect Mario's decisions at key milestones and analyze which biological neurons fired using FlyWire v1.0 annotations.
        </p>
      </div>

      <!-- Quick Metrics -->
      <div class="grid grid-cols-3 gap-3 w-full md:w-auto font-mono text-xs">
        <div class="bg-[var(--card-muted)]/70 border border-[var(--border-color)] rounded-xl p-3 text-center">
          <div class="text-[10px] text-[var(--text-dim)] uppercase">Total Neurons</div>
          <div class="text-base font-bold text-cyan-400 mt-0.5">30,000</div>
        </div>
        <div class="bg-[var(--card-muted)]/70 border border-[var(--border-color)] rounded-xl p-3 text-center">
          <div class="text-[10px] text-[var(--text-dim)] uppercase">Synapses</div>
          <div class="text-base font-bold text-emerald-400 mt-0.5">6,001,050</div>
        </div>
        <div class="bg-[var(--card-muted)]/70 border border-[var(--border-color)] rounded-xl p-3 text-center">
          <div class="text-[10px] text-[var(--text-dim)] uppercase">Motor DNs</div>
          <div class="text-base font-bold text-rose-400 mt-0.5">1,314</div>
        </div>
      </div>
    </header>

    <!-- Milestone Timeline Scrubber -->
    <section class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-2xl p-5 shadow-lg space-y-4 font-mono">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
        <div>
          <h2 class="text-sm font-bold text-white uppercase tracking-wider">
            Trajectory Timeline & Milestone Events
          </h2>
          <p class="text-xs text-[var(--text-muted)]">
            Click on key milestones or use the slider to inspect policy firing states
          </p>
        </div>
        <div class="flex items-center gap-2 text-xs">
          <button id="prevBtn" class="px-3.5 py-1.5 bg-[var(--card-muted)] hover:bg-[var(--border-hover)] border border-[var(--border-color)] rounded-lg text-gray-200 transition">
            ◀ Prev Frame
          </button>
          <span id="frameCounter" class="text-xs font-mono px-3 py-1.5 bg-black/50 rounded-lg border border-[var(--border-color)] text-cyan-300">
            Frame 1 / 35
          </span>
          <button id="nextBtn" class="px-3.5 py-1.5 bg-[var(--card-muted)] hover:bg-[var(--border-hover)] border border-[var(--border-color)] rounded-lg text-gray-200 transition">
            Next Frame ▶
          </button>
        </div>
      </div>

      <!-- Milestone Chips -->
      <div id="milestoneChips" class="flex flex-wrap gap-1.5 text-xs">
        <!-- Rendered by JS -->
      </div>

      <!-- Scrubber Slider -->
      <div class="space-y-1.5 pt-1">
        <input type="range" id="frameSlider" min="0" max="34" value="0" class="w-full h-2 bg-[var(--card-muted)] rounded-lg appearance-none cursor-pointer accent-cyan-500">
        <div class="flex justify-between text-[11px] text-[var(--text-muted)]">
          <span>Start (Step 1, x=40)</span>
          <span>Goomba 1 (Step 40)</span>
          <span>Pipe 4 (Step 150)</span>
          <span>Pit 1 (Step 225)</span>
          <span>Flagpole (Step 488, x=3161)</span>
        </div>
      </div>
    </section>

    <!-- Main Grid: Decision Card + 3 Brain Layers -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-5 font-mono">

      <!-- Left Panel: Current Decision & Mario Status -->
      <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-2xl p-5 shadow-lg space-y-4">
        <div class="border-b border-[var(--border-color)] pb-3">
          <span class="text-[10px] uppercase tracking-wider text-[var(--text-dim)] font-bold">Decision State</span>
          <h3 id="currentMilestoneText" class="text-base font-bold text-cyan-400 mt-1">
            Game Spawn & Visual Intake
          </h3>
        </div>

        <!-- Position & Critic Stats -->
        <div class="grid grid-cols-2 gap-3 text-xs">
          <div class="bg-[var(--card-muted)]/70 border border-[var(--border-color)] rounded-xl p-3">
            <span class="text-[10px] text-[var(--text-dim)] uppercase">Mario (X, Y)</span>
            <div id="marioCoords" class="text-base font-bold text-white mt-0.5">
              X=40, Y=93
            </div>
          </div>
          <div class="bg-[var(--card-muted)]/70 border border-[var(--border-color)] rounded-xl p-3">
            <span class="text-[10px] text-[var(--text-dim)] uppercase">Value V(s)</span>
            <div id="marioValue" class="text-base font-bold text-emerald-400 mt-0.5">
              +7.74
            </div>
          </div>
        </div>

        <!-- Action Badge -->
        <div class="bg-[var(--card-muted)]/70 border border-[var(--border-color)] rounded-xl p-3.5 space-y-1.5">
          <div class="flex justify-between items-center text-xs">
            <span class="text-[var(--text-muted)]">Selected Action:</span>
            <span id="actionBadge" class="px-2.5 py-1 bg-rose-500/20 text-rose-400 border border-rose-500/30 rounded-lg text-xs font-bold">
              RIGHT+A+B
            </span>
          </div>
          <div class="text-[11px] text-[var(--text-muted)]">
            Hoyer Sparsity L1/L2: <span id="sparsityVal" class="text-cyan-400 font-bold">111.5</span>
          </div>
        </div>

        <!-- Action Probabilities Bars -->
        <div class="space-y-2 text-xs">
          <div class="text-[10px] text-[var(--text-dim)] uppercase font-semibold">
            Actor Head Action Probabilities
          </div>
          <div id="actionBars" class="space-y-1.5">
            <!-- Rendered by JS -->
          </div>
        </div>
      </div>

      <!-- Right Panels (spans 2 cols): Functional Brain Superclasses -->
      <div class="lg:col-span-2 space-y-5">

        <!-- 3 Activity Meters -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-xl p-4 border-l-4 border-l-amber-400">
            <div class="text-xs text-[var(--text-muted)] font-medium">Visual Projections (9,201 VPs)</div>
            <div id="vpMean" class="text-xl font-bold text-amber-400 mt-1">7.720</div>
            <p class="text-[11px] text-[var(--text-dim)] mt-1">Mean sensory drive from CNN</p>
          </div>
          <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-xl p-4 border-l-4 border-l-cyan-400">
            <div class="text-xs text-[var(--text-muted)] font-medium">Central Brain (19,485 Core)</div>
            <div id="cbMean" class="text-xl font-bold text-cyan-400 mt-1">0.061</div>
            <p class="text-[11px] text-[var(--text-dim)] mt-1">Mean recurrent internal activation</p>
          </div>
          <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-xl p-4 border-l-4 border-l-rose-400">
            <div class="text-xs text-[var(--text-muted)] font-medium">Descending (1,314 DNs)</div>
            <div id="dnMean" class="text-xl font-bold text-rose-400 mt-1">0.097</div>
            <p class="text-[11px] text-[var(--text-dim)] mt-1">Mean motor command voltage</p>
          </div>
        </div>

        <!-- Biological Insights Card -->
        <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-xl p-4 text-xs text-[var(--text-muted)] space-y-1.5">
          <div class="font-bold text-cyan-400 text-sm flex items-center gap-2">
            Biological Circuit Alignment
          </div>
          <p>
            • <strong>Lobula Columnar Looming Detectors (LC10, LC12, LoVP):</strong> Visual neurons respond strongly to impending obstacles (Goombas and Pipe lips) entering Mario's field of view.
          </p>
          <p>
            • <strong>Giant Fiber Escape Takeoff Neurons (DNp01, DNp06, DNp31):</strong> Motor output neurons saturate at maximum firing magnitude (-1.000) during Mario's jump maneuvers.
          </p>
        </div>

        <!-- Interactive Neuron Body Inspector Table -->
        <div class="bg-[var(--card-bg)] border border-[var(--border-color)] rounded-2xl p-5 shadow-lg space-y-4">
          <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-2">
            <div>
              <h3 class="text-sm font-bold text-white uppercase tracking-wider">
                Top Active Neurons (FlyWire Annotations)
              </h3>
              <p class="text-xs text-[var(--text-muted)]">
                Extracted directly from MaleCNS v1.0 and 30k subgraph metadata
              </p>
            </div>

            <!-- Tab Filters -->
            <div class="flex items-center gap-1.5 bg-black/40 p-1 rounded-xl border border-[var(--border-color)] text-xs">
              <button class="tab-btn px-3 py-1 rounded-lg font-medium bg-cyan-600 text-white" data-tab="overall">
                Top 25
              </button>
              <button class="tab-btn px-3 py-1 rounded-lg font-medium text-[var(--text-muted)] hover:text-white" data-tab="dn">
                Motor (DNs)
              </button>
              <button class="tab-btn px-3 py-1 rounded-lg font-medium text-[var(--text-muted)] hover:text-white" data-tab="vp">
                Visual (VPs)
              </button>
            </div>
          </div>

          <!-- Search Box -->
          <div>
            <input type="text" id="neuronSearch" placeholder="Filter by Body ID, cell type (e.g. DNp, LC10, 10228)..." class="w-full bg-black/40 border border-[var(--border-color)] rounded-xl px-4 py-2 text-xs text-white focus:outline-none focus:border-cyan-500 placeholder-[var(--text-dim)]">
          </div>

          <!-- Table Container -->
          <div class="overflow-x-auto rounded-xl border border-[var(--border-color)] max-h-80 overflow-y-auto">
            <table class="w-full text-left text-xs">
              <thead class="bg-black/75 text-[var(--text-dim)] sticky top-0 uppercase tracking-wider text-[10px] border-b border-[var(--border-color)]">
                <tr>
                  <th class="px-3.5 py-2.5">Body ID</th>
                  <th class="px-3.5 py-2.5">Cell Type</th>
                  <th class="px-3.5 py-2.5">Superclass</th>
                  <th class="px-3.5 py-2.5">Side</th>
                  <th class="px-3.5 py-2.5">Neuromere</th>
                  <th class="px-3.5 py-2.5 text-right">Activation h(t)</th>
                </tr>
              </thead>
              <tbody id="neuronTableBody" class="divide-y divide-[var(--border-color)]">
                <!-- Rows rendered by JS -->
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </div>

  </div>

  <!-- Inlined Telemetry Data -->
  <script>
    const TELEMETRY_DATA = __TELEMETRY_DATA_JSON__;
    const ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B"];

    let currentFrameIdx = 0;
    let currentTab = "overall";
    let searchQuery = "";

    const slider = document.getElementById("frameSlider");
    const counter = document.getElementById("frameCounter");
    const milestoneText = document.getElementById("currentMilestoneText");
    const marioCoords = document.getElementById("marioCoords");
    const marioValue = document.getElementById("marioValue");
    const actionBadge = document.getElementById("actionBadge");
    const sparsityVal = document.getElementById("sparsityVal");
    const actionBars = document.getElementById("actionBars");

    const vpMean = document.getElementById("vpMean");
    const cbMean = document.getElementById("cbMean");
    const dnMean = document.getElementById("dnMean");
    const tableBody = document.getElementById("neuronTableBody");
    const searchInput = document.getElementById("neuronSearch");
    const chipContainer = document.getElementById("milestoneChips");

    // Initialize Milestone Chips
    TELEMETRY_DATA.frames.forEach((f, idx) => {
      if (idx === 0 || idx === 2 || idx === 6 || idx === 10 || idx === 16 || idx === 22 || idx === TELEMETRY_DATA.frames.length - 1) {
        const btn = document.createElement("button");
        btn.className = "px-2.5 py-1 rounded-lg border border-[var(--border-color)] bg-[var(--card-muted)] text-[var(--text-muted)] hover:border-cyan-500 hover:text-white transition font-mono text-[11px]";
        btn.textContent = `Step ${f.step}: ${f.milestone.replace("🏆 ", "").split("(")[0].trim()}`;
        btn.onclick = () => renderFrame(idx);
        chipContainer.appendChild(btn);
      }
    });

    function renderFrame(idx) {
      if (idx < 0) idx = 0;
      if (idx >= TELEMETRY_DATA.frames.length) idx = TELEMETRY_DATA.frames.length - 1;
      currentFrameIdx = idx;
      const f = TELEMETRY_DATA.frames[idx];

      slider.value = idx;
      counter.textContent = `Frame ${idx + 1} / ${TELEMETRY_DATA.frames.length}`;
      milestoneText.textContent = `Step ${f.step}: ${f.milestone.replace("🏆 ", "")}`;
      marioCoords.textContent = `X=${f.x_pos}, Y=${f.y_pos}`;
      marioValue.textContent = `${f.value >= 0 ? "+" : ""}${f.value.toFixed(2)}`;
      actionBadge.textContent = f.action_name;
      sparsityVal.textContent = f.sparsity_l1_l2.toFixed(1);

      vpMean.textContent = f.summary.vp_mean_abs.toFixed(3);
      cbMean.textContent = f.summary.cb_mean_abs.toFixed(3);
      dnMean.textContent = f.summary.dn_mean_abs.toFixed(3);

      // Render Action Bars
      actionBars.innerHTML = "";
      f.action_probs.forEach((p, aIdx) => {
        const isSelected = aIdx === f.action_idx;
        const row = document.createElement("div");
        row.className = "space-y-0.5";
        row.innerHTML = `
          <div class="flex justify-between text-[11px] ${isSelected ? "text-cyan-400 font-bold" : "text-gray-400"}">
            <span>${ACTION_NAMES[aIdx]}</span>
            <span>${(p * 100).toFixed(1)}%</span>
          </div>
          <div class="w-full bg-gray-900 h-1.5 rounded-full overflow-hidden">
            <div class="${isSelected ? "bg-cyan-500" : "bg-gray-700"} h-full" style="width: ${p * 100}%"></div>
          </div>
        `;
        actionBars.appendChild(row);
      });

      renderTable(f);
    }

    function renderTable(f) {
      let neurons = f.top_active_overall;
      if (currentTab === "dn") neurons = f.top_active_dn;
      else if (currentTab === "vp") neurons = f.top_active_vp;

      tableBody.innerHTML = "";
      neurons.forEach(n => {
        if (searchQuery) {
          const q = searchQuery.toLowerCase();
          const match = n.body_id.toString().includes(q) ||
                        n.type.toLowerCase().includes(q) ||
                        n.superclass.toLowerCase().includes(q) ||
                        n.neuromere.toLowerCase().includes(q);
          if (!match) return;
        }

        let badgeStyle = "bg-cyan-500/15 text-cyan-400 border-cyan-500/30";
        if (n.superclass === "descending_neuron") badgeStyle = "bg-rose-500/15 text-rose-400 border-rose-500/30";
        else if (n.superclass === "visual_projection") badgeStyle = "bg-amber-500/15 text-amber-400 border-amber-500/30";

        const tr = document.createElement("tr");
        tr.className = "hover:bg-slate-800/60 transition";
        tr.innerHTML = `
          <td class="px-3.5 py-2 font-bold text-white">${n.body_id}</td>
          <td class="px-3.5 py-2 text-gray-200">${n.type}</td>
          <td class="px-3.5 py-2">
            <span class="inline-block px-2 py-0.5 rounded text-[10px] border ${badgeStyle}">
              ${n.superclass}
            </span>
          </td>
          <td class="px-3.5 py-2 text-gray-300">${n.side}</td>
          <td class="px-3.5 py-2 text-gray-300">${n.neuromere}</td>
          <td class="px-3.5 py-2 text-right font-bold ${n.activation >= 0 ? "text-cyan-400" : "text-rose-400"}">
            ${n.activation.toFixed(3)}
          </td>
        `;
        tableBody.appendChild(tr);
      });
    }

    // Events
    document.getElementById("prevBtn").onclick = () => renderFrame(currentFrameIdx - 1);
    document.getElementById("nextBtn").onclick = () => renderFrame(currentFrameIdx + 1);
    slider.oninput = (e) => renderFrame(parseInt(e.target.value));

    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.onclick = () => {
        document.querySelectorAll(".tab-btn").forEach(b => {
          b.className = "tab-btn px-3 py-1 rounded-lg font-medium text-[var(--text-muted)] hover:text-white";
        });
        btn.className = "tab-btn px-3 py-1 rounded-lg font-medium bg-cyan-600 text-white";
        currentTab = btn.getAttribute("data-tab");
        renderTable(TELEMETRY_DATA.frames[currentFrameIdx]);
      };
    });

    searchInput.oninput = (e) => {
      searchQuery = e.target.value.trim();
      renderTable(TELEMETRY_DATA.frames[currentFrameIdx]);
    };

    renderFrame(0);
  </script>
</body>
</html>
"""


def main():
    print(f"Reading telemetry data from {DATA_JSON}...")
    with open(DATA_JSON, "r") as f:
        data_content = f.read()

    print("Injecting telemetry data into user-friendly template...")
    final_html = HTML_TEMPLATE.replace("__TELEMETRY_DATA_JSON__", data_content)

    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Saved local decision dashboard: {HTML_OUT} ({HTML_OUT.stat().st_size / 1024:.1f} KB)")

    ART_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(ART_OUT, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Saved artifact dashboard: {ART_OUT} ({ART_OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
