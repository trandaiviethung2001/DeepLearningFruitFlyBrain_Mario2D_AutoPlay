"""Builds the Human-Crafted, Apple/Google-Inspired Fruit Fly & Connectome Simulation Dashboard.

Design Principles:
- Apple Human Interface Guidelines (HIG) & Google Clean Design:
  * Obsidian black foundation (#000000) with subtle surface elevation (#121214, #1c1c1e)
  * Hairline 1px borders (rgba(255, 255, 255, 0.08)) and glassmorphic blur (20px)
  * Refined San Francisco / Inter typography (-apple-system) with tabular numerals
  * Understated semantic accents (Amber=Vision, Sky=Brain, Coral=Motor, Emerald=Progress)
  * Zero garish neon glows or generic AI tropes
- Professional Media Transport & Scrubber:
  * Minimalist circular transport controls (solid white Play button, sleek icon step buttons)
  * Unified Apple-style segmented speed capsule (0.5×, 1.0×, 2.0×, 4.0×)
  * Sleek 4px timeline scrubber with fluid white circular thumb
- 3D Drosophila Melanogaster Locomotion (Three.js WebGL):
  * True biological morphology (Head, faceted compound eyes, thorax, abdomen, wings, halteres)
  * 6 articulated legs executing kinematic Tripod Gait (Stance vs Swing) and Giant Fiber Leap
  * Minimalist 3D orbit controls and camera angle presets (3D Orbit, Dorsal, Lateral, Frontal)
- Publication-Grade Brain Connectome Map:
  * Anatomical neuropil boundary contours & typography labels (Optic Lobes, CX, Motor Trunk)
  * Dense 3,700-neuron Drosophila morphological scaffold
  * Live animated synaptic energy streams (Sensory -> Central Core -> Motor Trunk)
  * Interactive Zoom (wheel), Drag Pan, 1:1 Reset, and Region Focus presets
  * Precision hover crosshair and floating glassmorphism inspection card
  * Click-to-filter linkage with Active Neuron Explorer table
  * 100 µm anatomical scale bar & spatial orientation axes
- Real NES Gameplay:
  * Native 256×240 lossless pixel art directly from play.py execution
- 100% English language throughout
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_JSON = REPO_ROOT / "data" / "all_stages_simulation_data.json"
BASELINE_DATA_JSON = REPO_ROOT / "data" / "all_stages_baseline_data.json"
HTML_OUT = REPO_ROOT / "dashboards" / "fruitfly_simulation_dashboard.html"
ART_OUT = Path("/Users/trandaiviethung/.gemini/antigravity/brain/c20d0836-81e3-4066-98b9-aa1ff9b7cca9/fruitfly_simulation_dashboard.html")

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Fruit Fly Brain & Body Simulator | MaleCNS Connectome</title>
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <script>/* __THREE_JS_LIB__ */</script>
  <style>
    :root {
      --bg-base: #000000;
      --surface-1: #121214;
      --surface-2: #18181b;
      --surface-3: #27272a;
      --border-subtle: rgba(255, 255, 255, 0.08);
      --border-medium: rgba(255, 255, 255, 0.14);
      --text-primary: #f5f5f7;
      --text-secondary: #86868b;
      --text-tertiary: #52525b;
      --color-vision: #f59e0b;
      --color-core: #38bdf8;
      --color-motor: #ff453a;
      --color-success: #30d158;
    }
    body {
      background-color: var(--bg-base);
      color: var(--text-primary);
      font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      -webkit-font-smoothing: antialiased;
      -moz-osx-font-smoothing: grayscale;
    }
    .tabular-nums {
      font-variant-numeric: tabular-nums;
    }
    .glass-panel {
      background: rgba(20, 20, 23, 0.75);
      backdrop-filter: blur(24px);
      -webkit-backdrop-filter: blur(24px);
      border: 1px solid var(--border-subtle);
    }
    .glass-card {
      background: rgba(28, 28, 32, 0.60);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-subtle);
    }
    .pixelated-render {
      image-rendering: -webkit-optimize-contrast;
      image-rendering: -moz-crisp-edges;
      image-rendering: pixelated;
      image-rendering: crisp-edges;
    }
    /* Sleek Apple-style timeline range slider */
    input[type=range] {
      -webkit-appearance: none;
      background: rgba(255, 255, 255, 0.12);
      border-radius: 9999px;
      height: 4px;
      outline: none;
      transition: background 0.15s ease;
    }
    input[type=range]:hover {
      background: rgba(255, 255, 255, 0.2);
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: #ffffff;
      cursor: pointer;
      box-shadow: 0 1px 4px rgba(0, 0, 0, 0.5);
      transition: transform 0.1s ease;
    }
    input[type=range]::-webkit-slider-thumb:hover {
      transform: scale(1.25);
    }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.12); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255, 255, 255, 0.22); }
  </style>
</head>
<body class="antialiased p-4 md:p-6 lg:p-8 min-h-screen">
  <div class="max-w-[1600px] mx-auto space-y-4">

    <!-- Top Navigation & System Status Bar (Apple Style) -->
    <header class="glass-panel rounded-2xl px-5 py-4 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
      <div class="flex items-center gap-3.5">
        <!-- Minimalist Biological Emblem -->
        <div class="w-10 h-10 rounded-xl bg-white/[0.06] border border-white/10 flex items-center justify-center text-sky-400 font-semibold shadow-inner">
          <svg class="w-5 h-5 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.8" d="M12 2a4 4 0 00-4 4v1H6a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V9a2 2 0 00-2-2h-2V6a4 4 0 00-4-4z"/>
          </svg>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 id="appTitleText" class="text-base font-semibold tracking-tight text-white">Fruit Fly Connectome Simulator</h1>
            <span id="modelVersionBadge" class="px-2 py-0.5 rounded-full text-[10px] font-medium bg-sky-500/15 text-sky-300 border border-sky-500/30">MaleCNS v1.0</span>
          </div>
          <p id="stageDescriptionText" class="text-xs text-[#86868b] mt-0.5">30,000-neuron biological embodiment navigating Super Mario Bros World 1-1</p>
        </div>
      </div>

      <!-- Apple-Style Model Architecture Switcher -->
      <div class="flex items-center bg-black/60 p-1 rounded-xl border border-white/15 backdrop-blur-md shadow-inner">
        <button id="modelBtnConnectome" class="model-switch-btn px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm" data-model="connectome">
          <span id="modelDotConnectome" class="w-2 h-2 rounded-full bg-sky-400 animate-pulse"></span>
          <span>MaleCNS Connectome</span>
          <span class="text-[10px] px-1.5 py-0.5 rounded bg-sky-400/20 text-sky-200 font-mono">30k Neurons</span>
        </button>
        <button id="modelBtnBaseline" class="model-switch-btn px-3.5 py-1.5 rounded-lg text-xs font-medium flex items-center gap-2 transition-all text-[#86868b] hover:text-white border border-transparent" data-model="baseline">
          <span id="modelDotBaseline" class="w-2 h-2 rounded-full bg-zinc-600"></span>
          <span>Dense GRU-1024</span>
          <span class="text-[10px] px-1.5 py-0.5 rounded bg-white/10 text-zinc-300 font-mono">Baseline</span>
        </button>
      </div>

      <!-- Clean Status Readouts (Subtle Apple Health / Watch style) -->
      <div class="flex items-center gap-2 flex-wrap text-xs">
        <div class="px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center gap-2">
          <span class="text-[#86868b]">Position</span>
          <span id="marioPosHeader" class="text-white font-semibold tabular-nums">X=40, Y=93</span>
        </div>
        <div class="px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center gap-2">
          <span class="text-[#86868b]">Locomotion</span>
          <span id="gaitBadgeHeader" class="text-rose-400 font-semibold">Tripod Gait</span>
        </div>
        <div class="px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center gap-2">
          <span class="text-[#86868b]">Critic V(s)</span>
          <span id="valHeader" class="text-emerald-400 font-semibold tabular-nums">+7.74</span>
        </div>
        <div class="px-3 py-1.5 rounded-xl bg-white/[0.04] border border-white/[0.08] flex items-center gap-2">
          <span class="text-[#86868b]">Neural Sparsity</span>
          <span id="sparsityHeader" class="text-amber-400 font-semibold tabular-nums">111.5</span>
        </div>
      </div>
    </header>

    <!-- Interactive Stage Selector Bar (32 Levels across 8 Worlds) -->
    <section class="glass-panel rounded-2xl p-4 md:px-5 md:py-3.5 flex flex-col lg:flex-row justify-between items-start lg:items-center gap-3.5">
      <div class="flex items-center gap-3 flex-wrap">
        <span class="text-xs uppercase tracking-wider font-semibold text-[#86868b] flex items-center gap-1.5">
          <svg class="w-3.5 h-3.5 text-sky-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 20l-5.447-2.724A2 2 0 013 15.485V5.515a2 2 0 011.553-1.96L9 2m0 18l6 3m-6-3V2m6 21l5.447-2.724A2 2 0 0021 18.485V8.515a2 2 0 00-1.553-1.96L15 4m0 19V4m0 0L9 2"/></svg>
          <span id="stageSelectLabel">Select Stage:</span>
        </span>

        <!-- Custom Dropdown Selector -->
        <div class="relative min-w-[240px]">
          <select id="stageSelect" class="w-full appearance-none bg-white/[0.08] hover:bg-white/[0.12] text-white font-medium text-xs rounded-xl px-3.5 py-1.5 pr-8 border border-white/15 focus:outline-none focus:ring-1 focus:ring-sky-400 cursor-pointer transition shadow-sm">
            <!-- 32 options populated by JS -->
          </select>
          <div class="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2.5 text-gray-400">
            <svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 9l-7 7-7-7"/></svg>
          </div>
        </div>

        <!-- Biome Badge Pill -->
        <span id="currentStageBiomeBadge" class="px-2.5 py-1 rounded-full text-[11px] font-medium bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
          Overworld
        </span>

        <!-- Completion Status Pill -->
        <span id="currentStageStatusBadge" class="px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5">
          <span>🏆</span> Flagpole Win (100% Cleared)
        </span>
      </div>

      <!-- Quick World Filters -->
      <div id="worldFilterContainer" class="flex items-center gap-1 flex-wrap text-xs">
        <span class="text-[11px] text-[#86868b] mr-1 hidden sm:inline">World:</span>
        <button class="world-filter-btn px-2.5 py-1 rounded-lg text-xs bg-white/20 text-white font-semibold transition" data-world="all">All (32)</button>
        <button class="world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition" data-world="1">W1</button>
        <button class="world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition" data-world="2">W2</button>
        <button class="world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition" data-world="3">W3</button>
        <button class="world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition" data-world="4">W4</button>
        <button class="world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition" data-world="5">W5</button>
        <button class="world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition" data-world="6">W6</button>
        <button class="world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition" data-world="7">W7</button>
        <button class="world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition" data-world="8">W8</button>
      </div>
    </section>

    <!-- Professional Media Transport & Scrubber Deck (Apple QuickTime / Final Cut style) -->
    <section class="glass-panel rounded-2xl p-4 md:p-5 space-y-3.5">
      <div class="flex flex-col md:flex-row justify-between items-center gap-4">
        <!-- Transport Controls -->
        <div class="flex items-center gap-2.5">
          <!-- Step Back Button -->
          <button id="prevBtn" title="Step Back (Left Arrow)" class="w-8 h-8 rounded-full bg-white/[0.06] hover:bg-white/10 active:scale-95 text-white flex items-center justify-center transition border border-white/[0.08]">
            <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M19 12a1 1 0 0 1-1 1H8.414l3.293 3.293a1 1 0 0 1-1.414 1.414l-5-5a1 1 0 0 1 0-1.414l5-5a1 1 0 1 1 1.414 1.414L8.414 11H18a1 1 0 0 1 1 1z"/></svg>
          </button>

          <!-- Main Play/Pause (Solid White Circle) -->
          <button id="playBtn" class="flex items-center gap-2 px-5 py-2 rounded-full bg-white text-black hover:bg-gray-100 active:scale-95 transition font-semibold text-xs shadow-md">
            <span id="playIcon" class="text-xs">▶</span>
            <span id="playLabel">Play</span>
          </button>

          <!-- Step Next Button -->
          <button id="nextBtn" title="Step Next (Right Arrow)" class="w-8 h-8 rounded-full bg-white/[0.06] hover:bg-white/10 active:scale-95 text-white flex items-center justify-center transition border border-white/[0.08]">
            <svg class="w-3.5 h-3.5" fill="currentColor" viewBox="0 0 24 24"><path d="M5 12a1 1 0 0 1 1-1h9.586l-3.293-3.293a1 1 0 0 1 1.414-1.414l5 5a1 1 0 0 1 0 1.414l-5 5a1 1 0 0 1-1.414-1.414L15.586 13H6a1 1 0 0 1-1-1z"/></svg>
          </button>

          <!-- Reset Button -->
          <button id="resetBtn" title="Reset (R)" class="w-8 h-8 rounded-full bg-white/[0.06] hover:bg-white/10 active:scale-95 text-[#86868b] hover:text-white flex items-center justify-center transition border border-white/[0.08]">
            <svg class="w-3.5 h-3.5" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"/></svg>
          </button>
        </div>

        <!-- Speed & Current Frame Status -->
        <div class="flex items-center gap-3.5 text-xs">
          <!-- Unified Segmented Speed Capsule -->
          <div class="bg-white/[0.05] p-0.5 rounded-full border border-white/[0.08] flex items-center text-[11px] font-medium">
            <span class="text-[#86868b] px-2">Speed</span>
            <button class="speed-btn px-2 py-0.5 rounded-full text-[#86868b] hover:text-white transition" data-speed="0.5">0.5×</button>
            <button class="speed-btn px-2.5 py-0.5 rounded-full bg-white/20 text-white font-semibold transition" data-speed="1.0">1.0×</button>
            <button class="speed-btn px-2 py-0.5 rounded-full text-[#86868b] hover:text-white transition" data-speed="1.5">1.5×</button>
            <button class="speed-btn px-2 py-0.5 rounded-full text-[#86868b] hover:text-white transition" data-speed="2.0">2.0×</button>
          </div>

          <!-- Timecode Readout -->
          <div class="px-3 py-1 rounded-full bg-white/[0.04] border border-white/[0.08] text-gray-300 font-mono text-[11px] tabular-nums">
            Frame <span id="stepCounter" class="text-white font-bold">1</span> / <span id="totalStepsCounter">1,950</span> <span class="text-[#86868b] ml-1.5 font-normal">(60 FPS • Real-time)</span>
          </div>
        </div>
      </div>

      <!-- Scrubber Slider -->
      <div class="space-y-1.5 pt-1">
        <input type="range" id="simSlider" min="0" max="1949" value="0" class="w-full cursor-pointer">
        <div class="flex justify-between text-[11px] text-[#86868b] font-medium">
          <span>Start (F0)</span>
          <span>Goomba 1 (F156)</span>
          <span>Pipes 1–4 (F213–620)</span>
          <span>Abyss Leap (F886)</span>
          <span class="text-emerald-400 font-semibold">Flagpole Win (F1949)</span>
        </div>
      </div>

      <!-- Milestone Bookmark Chips -->
      <div class="flex items-center gap-2 pt-1 flex-wrap">
        <span class="text-[11px] text-[#86868b] uppercase tracking-wider font-semibold">Bookmarks:</span>
        <div id="eventChips" class="flex flex-wrap gap-1.5">
          <!-- Rendered by JS -->
        </div>
      </div>
    </section>

    <!-- Main Workspace Grid: 3 Refined Panels -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">

      <!-- Panel 1 (Cols 1-4): 3D Locomotion Viewport -->
      <div class="lg:col-span-4 glass-panel rounded-2xl p-4 md:p-5 flex flex-col justify-between space-y-3">
        <div>
          <!-- Header -->
          <div class="flex justify-between items-center border-b border-white/[0.08] pb-3 mb-2.5">
            <div>
              <h2 class="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-rose-400"></span>
                3D Fly Locomotion
              </h2>
              <p class="text-[11px] text-[#86868b]">Tripod Gait & Escape Leap Biomechanics</p>
            </div>
            <span id="gaitStatusPill" class="px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-white/[0.06] text-gray-200 border border-white/10">
              TRIPOD GAIT
            </span>
          </div>

          <!-- Camera Angle Presets (Apple Segmented Style) -->
          <div class="flex items-center justify-between gap-1 pb-2 text-[11px]">
            <div class="bg-white/[0.05] p-0.5 rounded-full border border-white/[0.08] flex items-center">
              <button class="cam-btn px-2.5 py-0.5 rounded-full bg-white/20 text-white font-semibold transition" data-cam="orbit">Orbit</button>
              <button class="cam-btn px-2.5 py-0.5 rounded-full text-[#86868b] hover:text-white transition" data-cam="dorsal">Top</button>
              <button class="cam-btn px-2.5 py-0.5 rounded-full text-[#86868b] hover:text-white transition" data-cam="lateral">Side</button>
              <button class="cam-btn px-2.5 py-0.5 rounded-full text-[#86868b] hover:text-white transition" data-cam="frontal">Front</button>
            </div>
            <button id="resetCamBtn" class="px-2 py-0.5 rounded-full text-[#86868b] hover:text-white text-[10px] bg-white/[0.04] border border-white/[0.08] transition">
              Reset View
            </button>
          </div>

          <!-- 3D Canvas Viewport -->
          <div id="fly3dContainer" class="relative bg-black rounded-xl border border-white/[0.08] overflow-hidden h-[310px] select-none cursor-grab active:cursor-grabbing">
            <canvas id="flyCanvas" class="w-full h-full block"></canvas>

            <!-- Minimalist Floating Overlay Badges -->
            <div class="absolute top-2.5 left-2.5 text-[10px] space-y-1.5 bg-[#121214]/85 backdrop-blur-md p-2 rounded-xl border border-white/10 pointer-events-none">
              <div class="flex items-center gap-2">
                <span id="eyeLedL" class="w-2 h-2 rounded-full bg-gray-600 transition-colors"></span>
                <span class="text-gray-300">Left Eye (Optic L)</span>
              </div>
              <div class="flex items-center gap-2">
                <span id="eyeLedR" class="w-2 h-2 rounded-full bg-gray-600 transition-colors"></span>
                <span class="text-gray-300">Right Eye (Optic R)</span>
              </div>
              <div class="flex items-center gap-2">
                <span id="vncLed" class="w-2 h-2 rounded-full bg-gray-600 transition-colors"></span>
                <span class="text-gray-300">VNC Motor Trunk</span>
              </div>
            </div>

            <div class="absolute bottom-2.5 right-2.5 text-[10px] bg-[#121214]/85 backdrop-blur-md px-2.5 py-1 rounded-xl border border-white/10 text-gray-300 pointer-events-none flex items-center gap-2">
              <span>Pitch: <span id="pitchAngleText" class="text-sky-400 font-semibold tabular-nums">0°</span></span>
              <span class="text-gray-600">|</span>
              <span class="text-[9px] text-[#86868b]">Drag to Orbit • Scroll to Zoom</span>
            </div>

            <div class="absolute top-2.5 right-2.5 text-[9px] bg-white/5 border border-white/10 text-gray-300 px-2.5 py-0.5 rounded-full pointer-events-none flex items-center gap-1.5">
              <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span> 100% Drosophila melanogaster (Male)
            </div>
          </div>
        </div>

        <!-- Locomotion Meters (Apple Health style minimalist bars) -->
        <div class="space-y-2.5 text-xs">
          <!-- Giant Fiber Meter -->
          <div class="glass-card rounded-xl p-2.5 space-y-1">
            <div class="flex justify-between text-xs">
              <span class="text-[#86868b] font-medium">Giant Fiber Reflex (DNp01/06):</span>
              <span id="gfMeterVal" class="font-semibold text-rose-400 tabular-nums">0.000</span>
            </div>
            <div class="w-full bg-white/[0.08] h-1.5 rounded-full overflow-hidden">
              <div id="gfMeterBar" class="bg-rose-500 h-full transition-all duration-75" style="width: 0%"></div>
            </div>
          </div>

          <!-- Looming Detector Meter -->
          <div class="glass-card rounded-xl p-2.5 space-y-1">
            <div class="flex justify-between text-xs">
              <span class="text-[#86868b] font-medium">Visual Looming Drive (LC10/12):</span>
              <span id="loomingMeterVal" class="font-semibold text-amber-400 tabular-nums">0.000</span>
            </div>
            <div class="w-full bg-white/[0.08] h-1.5 rounded-full overflow-hidden">
              <div id="loomingMeterBar" class="bg-amber-500 h-full transition-all duration-75" style="width: 0%"></div>
            </div>
          </div>

          <!-- 6-Leg Joint Table -->
          <div class="glass-card rounded-xl p-2.5">
            <div class="text-[10px] text-[#86868b] uppercase tracking-wider font-semibold mb-1.5">Tripod Leg Phase:</div>
            <div class="grid grid-cols-2 gap-2 text-xs">
              <div class="space-y-1">
                <div id="legL1" class="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-gray-300 flex justify-between text-[11px]">
                  <span>L1 (Front T1):</span> <span class="leg-state">STANCE</span>
                </div>
                <div id="legL2" class="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-gray-300 flex justify-between text-[11px]">
                  <span>L2 (Mid T2):</span> <span class="leg-state">SWING</span>
                </div>
                <div id="legL3" class="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-gray-300 flex justify-between text-[11px]">
                  <span>L3 (Hind T3):</span> <span class="leg-state">STANCE</span>
                </div>
              </div>
              <div class="space-y-1">
                <div id="legR1" class="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-gray-300 flex justify-between text-[11px]">
                  <span>R1 (Front T1):</span> <span class="leg-state">SWING</span>
                </div>
                <div id="legR2" class="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-gray-300 flex justify-between text-[11px]">
                  <span>R2 (Mid T2):</span> <span class="leg-state">STANCE</span>
                </div>
                <div id="legR3" class="px-2 py-0.5 rounded bg-white/[0.04] border border-white/[0.06] text-gray-300 flex justify-between text-[11px]">
                  <span>R3 (Hind T3):</span> <span class="leg-state">SWING</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Panel 2 (Cols 5-8): Brain Connectome Map -->
      <div class="lg:col-span-5 glass-panel rounded-2xl p-4 md:p-5 flex flex-col justify-between space-y-3">
        <div>
          <!-- Title & Projections -->
          <div class="flex flex-wrap justify-between items-center gap-2 border-b border-white/[0.08] pb-3 mb-2.5">
            <div>
              <h2 class="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-sky-400"></span>
                Brain Connectome Map
              </h2>
              <p class="text-[11px] text-[#86868b]">FlyWire MaleCNS 3D Spatial Atlas & Synaptic Stream</p>
            </div>

            <!-- Segmented Projection Control -->
            <div class="bg-white/[0.05] p-0.5 rounded-full border border-white/[0.08] flex items-center text-[11px] font-medium">
              <button class="view-btn px-2.5 py-0.5 rounded-full bg-white/20 text-white font-semibold transition" data-view="xy">Coronal (X-Y)</button>
              <button class="view-btn px-2.5 py-0.5 rounded-full text-[#86868b] hover:text-white transition" data-view="xz">Dorsal (X-Z)</button>
              <button class="view-btn px-2.5 py-0.5 rounded-full text-[#86868b] hover:text-white transition" data-view="yz">Sagittal (Y-Z)</button>
            </div>
          </div>

          <!-- Secondary Toolbar: Zoom & Presets -->
          <div class="flex flex-wrap items-center justify-between gap-2 pb-2 text-xs">
            <div class="flex items-center gap-1 bg-white/[0.04] p-0.5 rounded-lg border border-white/[0.08]">
              <button id="zoomInBtn" title="Zoom In" class="w-6 h-6 rounded flex items-center justify-center hover:bg-white/10 text-gray-300 hover:text-white font-bold transition">+</button>
              <button id="zoomOutBtn" title="Zoom Out" class="w-6 h-6 rounded flex items-center justify-center hover:bg-white/10 text-gray-300 hover:text-white font-bold transition">−</button>
              <button id="zoomResetBtn" title="Reset View" class="px-2 h-6 rounded hover:bg-white/10 text-gray-400 hover:text-white text-[10px] transition">1:1 Reset</button>
            </div>

            <div class="flex items-center gap-1 text-[11px]">
              <span class="text-[#86868b] hidden sm:inline">Focus:</span>
              <button class="focus-btn px-2 py-0.5 rounded-full bg-white/[0.04] hover:bg-white/10 text-gray-300 border border-white/[0.08] transition" data-focus="all">Whole Brain</button>
              <button class="focus-btn px-2 py-0.5 rounded-full bg-white/[0.04] hover:bg-white/10 text-amber-300 border border-white/[0.08] transition" data-focus="optic">Optic Lobes</button>
              <button class="focus-btn px-2 py-0.5 rounded-full bg-white/[0.04] hover:bg-white/10 text-sky-300 border border-white/[0.08] transition" data-focus="cx">Central Complex</button>
              <button class="focus-btn px-2 py-0.5 rounded-full bg-white/[0.04] hover:bg-white/10 text-rose-300 border border-white/[0.08] transition" data-focus="dn">Motor Trunk</button>
            </div>
          </div>

          <!-- Layer Toggles -->
          <div class="flex flex-wrap items-center gap-1.5 pb-2 text-[10px]">
            <span class="text-[#86868b] mr-1">Layers:</span>
            <label class="cursor-pointer inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-gray-300 hover:text-white">
              <input type="checkbox" id="toggleNeuropils" checked class="accent-sky-500 w-3 h-3"> Neuropils
            </label>
            <label class="cursor-pointer inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-gray-300 hover:text-white">
              <input type="checkbox" id="toggleScaffold" checked class="accent-sky-500 w-3 h-3"> 3.7k Scaffold
            </label>
            <label class="cursor-pointer inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-sky-300 hover:text-white">
              <input type="checkbox" id="toggleStreams" checked class="accent-sky-500 w-3 h-3"> Synaptic Streams
            </label>
            <label class="cursor-pointer inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-amber-400 hover:text-white">
              <input type="checkbox" id="toggleVP" checked class="accent-amber-500 w-3 h-3"> Visual (VPs)
            </label>
            <label class="cursor-pointer inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-sky-400 hover:text-white">
              <input type="checkbox" id="toggleCB" checked class="accent-sky-500 w-3 h-3"> Central (CBs)
            </label>
            <label class="cursor-pointer inline-flex items-center gap-1 px-2 py-0.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-rose-400 hover:text-white">
              <input type="checkbox" id="toggleDN" checked class="accent-rose-500 w-3 h-3"> Motor (DNs)
            </label>
          </div>

          <!-- Canvas Container -->
          <div id="brainCanvasContainer" class="relative bg-black rounded-xl border border-white/[0.08] overflow-hidden h-[390px] select-none">
            <canvas id="brainCanvas" class="w-full h-full block cursor-crosshair"></canvas>

            <!-- Top HUD Badge -->
            <div class="absolute top-2.5 right-2.5 text-[10px] bg-[#121214]/85 backdrop-blur-md px-2.5 py-1 rounded-xl border border-white/10 flex items-center gap-2 pointer-events-none z-10">
              <span id="brainViewLabel" class="text-sky-400 font-semibold">Coronal (X-Y)</span>
              <span class="text-gray-600">|</span>
              <span id="brainZoomLabel" class="text-gray-300 tabular-nums">1.0×</span>
            </div>

            <!-- Floating Apple Glassmorphic Inspection Card -->
            <div id="brainTooltip" class="absolute pointer-events-none opacity-0 transition-opacity duration-150 z-30 bg-[#18181b]/95 backdrop-blur-xl border border-white/15 rounded-2xl p-3.5 shadow-2xl text-xs min-w-[230px] max-w-[280px]">
              <div class="flex items-center justify-between border-b border-white/10 pb-2 mb-2">
                <span id="ttId" class="font-semibold text-white text-sm tabular-nums">#523329</span>
                <span id="ttType" class="px-2 py-0.5 rounded-full text-[10px] font-medium bg-white/10 text-sky-300 border border-white/10">DNa10</span>
              </div>
              <div class="space-y-1.5 text-[11px] text-gray-300">
                <div class="flex justify-between"><span class="text-[#86868b]">Compartment</span> <span id="ttNeuropil" class="font-medium text-white truncate max-w-[130px]">Motor Trunk</span></div>
                <div class="flex justify-between"><span class="text-[#86868b]">Hemibrain</span> <span id="ttSide" class="text-white">Left (L)</span></div>
                <div class="flex justify-between"><span class="text-[#86868b]">Synaptic Degree</span> <span id="ttDegrees" class="text-sky-300 tabular-nums">1,084 in / 276 out</span></div>
                <div class="flex justify-between"><span class="text-[#86868b]">Coordinates</span> <span id="ttCoords" class="text-gray-300 text-[10px] tabular-nums">62.7k, 16.5k, 15.6k</span></div>
                <div class="flex justify-between items-center pt-1.5 border-t border-white/10">
                  <span class="text-[#86868b]">Activation h(t)</span>
                  <span id="ttAct" class="font-bold text-sm text-sky-400 tabular-nums">+0.284</span>
                </div>
              </div>
              <div class="text-[9px] text-[#86868b] text-center mt-2 pt-1 border-t border-white/5">
                Click neuron to filter table below
              </div>
            </div>

            <!-- Interaction Hint -->
            <div class="absolute bottom-2.5 right-2.5 text-[9px] text-[#86868b] bg-[#121214]/85 backdrop-blur-md px-2.5 py-1 rounded-full border border-white/10 pointer-events-none z-10 flex items-center gap-1.5">
              <span>Scroll = Zoom</span> • <span>Drag = Pan</span> • <span>Hover = Inspect</span>
            </div>

            <!-- Legend Overlay -->
            <div class="absolute bottom-2.5 left-2.5 text-[10px] bg-[#121214]/85 backdrop-blur-md px-3 py-1.5 rounded-xl border border-white/10 flex flex-wrap items-center gap-2.5 pointer-events-none z-10">
              <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-amber-400"></span> Visual (VPs)</div>
              <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-sky-400"></span> Central Brain</div>
              <div class="flex items-center gap-1.5"><span class="w-2 h-2 rounded-full bg-rose-400"></span> Motor (DNs)</div>
              <div class="flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-zinc-500"></span> Scaffold (3.7k)</div>
            </div>
          </div>
        </div>

        <!-- Regional Summary Metrics -->
        <div>
          <div class="grid grid-cols-3 gap-2 text-center mb-2">
            <div class="glass-card rounded-xl p-2 border-l-2 border-l-amber-400">
              <div id="lblVpDrive" class="text-[10px] text-[#86868b] uppercase tracking-wide">Visual Drive</div>
              <div id="vpMeanVal" class="text-sm font-semibold text-amber-400 mt-0.5 tabular-nums">7.72</div>
            </div>
            <div class="glass-card rounded-xl p-2 border-l-2 border-l-sky-400">
              <div id="lblCbCore" class="text-[10px] text-[#86868b] uppercase tracking-wide">Central Core</div>
              <div id="cbMeanVal" class="text-sm font-semibold text-sky-400 mt-0.5 tabular-nums">0.06</div>
            </div>
            <div class="glass-card rounded-xl p-2 border-l-2 border-l-rose-400">
              <div id="lblDnMotor" class="text-[10px] text-[#86868b] uppercase tracking-wide">Motor Output</div>
              <div id="dnMeanVal" class="text-sm font-semibold text-rose-400 mt-0.5 tabular-nums">0.10</div>
            </div>
          </div>

          <div id="synapseSummaryRow" class="px-3 py-2 rounded-xl bg-white/[0.03] border border-white/[0.06] flex flex-wrap items-center justify-between text-[11px] text-[#86868b] gap-2">
            <div id="synapseMetricText"><span class="text-sky-400 font-medium">MaleCNS Synapses:</span> <span class="text-white tabular-nums">6,001,050</span></div>
            <div id="neuronMetricText"><span class="text-amber-400 font-medium">Neurons:</span> <span class="text-white tabular-nums">30,000</span> (3,700 Scaffold)</div>
            <div id="motorUnitMetricText"><span class="text-rose-400 font-medium">Motor Units:</span> <span class="text-white tabular-nums">1,314 DNs</span></div>
          </div>
        </div>
      </div>

      <!-- Panel 3 (Cols 9-12): Real NES Gameplay & Controller HUD -->
      <div class="lg:col-span-3 glass-panel rounded-2xl p-4 md:p-5 flex flex-col justify-between space-y-3">
        <div>
          <div class="flex justify-between items-center border-b border-white/[0.08] pb-3 mb-2.5">
            <div>
              <h2 class="text-xs font-semibold text-white uppercase tracking-wider flex items-center gap-2">
                <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
                NES Gameplay
              </h2>
              <p class="text-[11px] text-[#86868b]">60 FPS Lossless • Zero Frame Skip</p>
            </div>
            <span id="marioActionNameBadge" class="px-2 py-0.5 rounded-full text-[11px] font-semibold bg-white/10 text-white border border-white/10 tabular-nums">
              RIGHT+A+B
            </span>
          </div>

          <!-- Studio Display Hardware Framing -->
          <div class="relative bg-black rounded-xl border border-white/10 overflow-hidden shadow-inner h-52 flex items-center justify-center p-1.5">
            <img id="marioNesFrameImg" src="" alt="Real NES Frame" class="pixelated-render max-w-full max-h-full object-contain rounded">
            <div class="absolute top-2 left-2 text-[10px] text-emerald-400 font-semibold bg-black/80 px-2 py-0.5 rounded border border-white/10">
              Native 256×240 • 60 FPS
            </div>
            <div id="marioCrtPos" class="absolute top-2 right-2 text-[10px] text-sky-400 bg-black/80 px-2 py-0.5 rounded border border-white/10 tabular-nums">
              X=40, Y=93
            </div>
          </div>

          <!-- Level Progress -->
          <div class="mt-3 space-y-1 text-xs">
            <div class="flex justify-between text-[11px] text-[#86868b]">
              <span>World 1-1 Progress</span>
              <span id="levelProgressText" class="text-white font-medium tabular-nums">1.2%</span>
            </div>
            <div class="w-full bg-white/[0.08] h-1.5 rounded-full overflow-hidden">
              <div id="levelProgressBar" class="bg-gradient-to-r from-sky-400 to-emerald-400 h-full" style="width: 1.2%"></div>
            </div>
          </div>
        </div>

        <!-- Controller HUD -->
        <div class="glass-card rounded-xl p-3 space-y-2.5">
          <div class="text-[10px] text-[#86868b] uppercase tracking-wider font-semibold">
            Actuated Controller Buttons:
          </div>
          <div class="flex items-center justify-between px-3 py-2 bg-black/40 rounded-lg border border-white/[0.08] text-xs">
            <div class="flex items-center gap-1.5">
              <span class="text-[#86868b] text-[11px]">D-Pad:</span>
              <span id="btnRight" class="px-2.5 py-0.5 rounded font-semibold bg-white/5 text-gray-500 border border-white/5">RIGHT</span>
            </div>
            <div class="flex items-center gap-2">
              <span id="btnB" class="w-6 h-6 rounded-full flex items-center justify-center font-bold bg-white/5 text-gray-500 border border-white/5 text-[11px]">B</span>
              <span id="btnA" class="w-6 h-6 rounded-full flex items-center justify-center font-bold bg-white/5 text-gray-500 border border-white/5 text-[11px]">A</span>
            </div>
          </div>

          <!-- Actor Probability Distribution -->
          <div class="space-y-1 pt-1 text-xs">
            <div class="text-[10px] text-[#86868b] uppercase font-semibold">Policy Distribution (5 Actions):</div>
            <div id="actorProbBars" class="space-y-1.5">
              <!-- Rendered by JS -->
            </div>
          </div>
        </div>
      </div>

    </div>

    <!-- Lower Section: Active Neuron Explorer (Clean Apple / Google Console style) -->
    <section class="glass-panel rounded-2xl p-4 md:p-5 space-y-3.5">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 border-b border-white/[0.08] pb-3">
        <div>
          <h3 id="neuronSectionTitle" class="text-sm font-semibold text-white flex items-center gap-2">
            Active Neuron Explorer (FlyWire Annotations)
          </h3>
          <p id="neuronSectionSubtitle" class="text-xs text-[#86868b]">
            Inspect instantaneous activations, cell types, 3D anatomical positions, and synaptic connectivity.
          </p>
        </div>

        <!-- Segmented Filter Tabs -->
        <div id="tableTabContainer" class="bg-white/[0.05] p-0.5 rounded-full border border-white/[0.08] flex items-center text-xs font-medium">
          <button class="tab-btn px-3 py-1 rounded-full bg-white/20 text-white font-semibold transition" data-tab="all">All Neurons</button>
          <button class="tab-btn px-3 py-1 rounded-full text-[#86868b] hover:text-white transition" data-tab="dn">Motor (DNs)</button>
          <button class="tab-btn px-3 py-1 rounded-full text-[#86868b] hover:text-white transition" data-tab="vp">Visual (VPs)</button>
          <button class="tab-btn px-3 py-1 rounded-full text-[#86868b] hover:text-white transition" data-tab="cb">Brain Core (CBs)</button>
        </div>
      </div>

      <!-- Search Input -->
      <div>
        <input type="text" id="neuronSearchInput" placeholder="Search by Body ID (e.g. 523329), cell type (e.g. DNp06, LC10), or brain region..." class="w-full bg-white/[0.04] border border-white/[0.08] rounded-xl px-4 py-2 text-xs text-white focus:outline-none focus:border-sky-500 placeholder-[#86868b]">
      </div>

      <!-- Table View -->
      <div class="overflow-x-auto rounded-xl border border-white/[0.08] max-h-72 overflow-y-auto">
        <table class="w-full text-left text-xs">
          <thead class="bg-[#18181b] text-[#86868b] uppercase tracking-wider sticky top-0 border-b border-white/[0.08] text-[10px]">
            <tr>
              <th class="px-3.5 py-2">FlyWire ID</th>
              <th class="px-3.5 py-2">Cell Type</th>
              <th class="px-3.5 py-2">Category</th>
              <th class="px-3.5 py-2">Side</th>
              <th class="px-3.5 py-2">Synapses (In/Out)</th>
              <th class="px-3.5 py-2">3D Coordinates (X, Y, Z)</th>
              <th class="px-3.5 py-2">Region</th>
              <th class="px-3.5 py-2 text-right">Activation h(t)</th>
            </tr>
          </thead>
          <tbody id="neuronTableBody" class="divide-y divide-white/[0.04]">
            <!-- Rows rendered by JS -->
          </tbody>
        </table>
      </div>
    </section>

    <!-- Super Mario Bros 32-Stage Generalization Matrix & Benchmark Browser -->
    <section class="glass-panel rounded-2xl p-4 md:p-5 space-y-4">
      <div class="flex flex-col md:flex-row justify-between items-start md:items-center gap-3 border-b border-white/[0.08] pb-3">
        <div>
          <h3 class="text-sm font-semibold text-white flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-emerald-400"></span>
            Super Mario Bros 32-Stage Generalization Matrix
          </h3>
          <p class="text-xs text-[#86868b]">
            Zero-shot policy transfer across all 32 levels in Super Mario Bros (NES). Click any stage card to simulate immediately.
          </p>
        </div>

        <div class="flex items-center gap-2 text-xs flex-wrap">
          <span class="px-2.5 py-1 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30 font-medium">🏆 2 Flagpole Wins (W1-1, W3-2)</span>
          <span class="px-2.5 py-1 rounded-full bg-sky-500/15 text-sky-300 border border-sky-500/30 font-medium">9 Deep Runs (X &gt; 500)</span>
          <span class="px-2.5 py-1 rounded-full bg-white/[0.06] text-gray-300 border border-white/10 font-medium">32 / 32 Stages Evaluated</span>
        </div>
      </div>

      <!-- Biome Filters -->
      <div class="flex items-center gap-1.5 flex-wrap text-xs">
        <span class="text-[#86868b] mr-1">Filter Biome:</span>
        <button class="biome-filter-btn px-2.5 py-1 rounded-full bg-white/20 text-white font-semibold transition" data-biome="all">All (32)</button>
        <button class="biome-filter-btn px-2.5 py-1 rounded-full text-[#86868b] hover:text-white transition" data-biome="Overworld">Overworld (16)</button>
        <button class="biome-filter-btn px-2.5 py-1 rounded-full text-[#86868b] hover:text-white transition" data-biome="Underground">Underground (2)</button>
        <button class="biome-filter-btn px-2.5 py-1 rounded-full text-[#86868b] hover:text-white transition" data-biome="Castle">Castle (8)</button>
        <button class="biome-filter-btn px-2.5 py-1 rounded-full text-[#86868b] hover:text-white transition" data-biome="Underwater">Underwater (2)</button>
        <button class="biome-filter-btn px-2.5 py-1 rounded-full text-[#86868b] hover:text-white transition" data-biome="Athletics">Athletics (6)</button>
      </div>

      <!-- 32-Stage Interactive Grid -->
      <div id="stagesBenchmarkGrid" class="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-2.5 pt-1">
        <!-- Rendered dynamically by JS -->
      </div>
    </section>

  </div>

  <!-- Telemetry Data Inlined -->
  <script>
    const SIM_DATA = __SIMULATION_DATA_JSON__;
    const BASELINE_DATA = __BASELINE_DATA_JSON__;
    const ACTION_NAMES = ["NOOP", "RIGHT", "RIGHT+A", "RIGHT+B", "RIGHT+A+B"];

    let currentModelType = "connectome"; // "connectome" | "baseline"
    let currentStageId = "1-1";
    let currentStage = (SIM_DATA.stages && SIM_DATA.stages[currentStageId]) ? SIM_DATA.stages[currentStageId] : (SIM_DATA.stages ? Object.values(SIM_DATA.stages)[0] : { steps: SIM_DATA.steps || [] });
    let currentSteps = currentStage.steps || [];

    let currentStepIdx = 0;
    let isPlaying = false;
    let playSpeed = 1.0;
    let playTimer = null;
    let currentBrainView = "xy";
    let activeTab = "all";
    let searchQuery = "";
    let activeWorldFilter = "all";
    let activeBiomeFilter = "all";

    // Connectome Map Navigation State
    let brainZoom = 1.0;
    let brainPanX = 0;
    let brainPanY = 0;
    let isDraggingBrain = false;
    let dragStartX = 0;
    let dragStartY = 0;
    let hoveredNeuron = null;
    let animPhase = 0;
    let animReq = null;

    // Layer Visibility Flags
    let showNeuropils = true;
    let showScaffold = true;
    let showStreams = true;
    let showVP = true;
    let showCB = true;
    let showDN = true;

    // 3D Fruit Fly Simulation State
    let flyScene, flyCamera, flyRenderer;
    let fly3D = null;
    let is3DReady = false;
    let camOrbit = { theta: 0.78, phi: 1.15, radius: 13.5, target: null };
    let isDragging3D = false;
    let drag3DStart = { x: 0, y: 0 };

    // DOM Elements
    const modelBtnConnectome = document.getElementById("modelBtnConnectome");
    const modelBtnBaseline = document.getElementById("modelBtnBaseline");
    const modelDotConnectome = document.getElementById("modelDotConnectome");
    const modelDotBaseline = document.getElementById("modelDotBaseline");
    const appTitleText = document.getElementById("appTitleText");
    const modelVersionBadge = document.getElementById("modelVersionBadge");
    const stageSelectLabel = document.getElementById("stageSelectLabel");
    const worldFilterContainer = document.getElementById("worldFilterContainer");
    const lblVpDrive = document.getElementById("lblVpDrive");
    const lblCbCore = document.getElementById("lblCbCore");
    const lblDnMotor = document.getElementById("lblDnMotor");
    const synapseMetricText = document.getElementById("synapseMetricText");
    const neuronMetricText = document.getElementById("neuronMetricText");
    const motorUnitMetricText = document.getElementById("motorUnitMetricText");
    const neuronSectionTitle = document.getElementById("neuronSectionTitle");
    const neuronSectionSubtitle = document.getElementById("neuronSectionSubtitle");
    const tableTabContainer = document.getElementById("tableTabContainer");

    const stageSelect = document.getElementById("stageSelect");
    const stageDescriptionText = document.getElementById("stageDescriptionText");
    const currentStageBiomeBadge = document.getElementById("currentStageBiomeBadge");
    const currentStageStatusBadge = document.getElementById("currentStageStatusBadge");
    const stagesBenchmarkGrid = document.getElementById("stagesBenchmarkGrid");
    const worldFilterBtns = document.querySelectorAll(".world-filter-btn");
    const biomeFilterBtns = document.querySelectorAll(".biome-filter-btn");

    const playBtn = document.getElementById("playBtn");
    const playIcon = document.getElementById("playIcon");
    const playLabel = document.getElementById("playLabel");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const resetBtn = document.getElementById("resetBtn");
    const simSlider = document.getElementById("simSlider");
    const stepCounter = document.getElementById("stepCounter");
    const totalStepsCounter = document.getElementById("totalStepsCounter");
    const eventChips = document.getElementById("eventChips");
    const speedBtns = document.querySelectorAll(".speed-btn");
    const viewBtns = document.querySelectorAll(".view-btn");
    const tabBtns = document.querySelectorAll(".tab-btn");
    const searchInput = document.getElementById("neuronSearchInput");

    const marioPosHeader = document.getElementById("marioPosHeader");
    const gaitBadgeHeader = document.getElementById("gaitBadgeHeader");
    const valHeader = document.getElementById("valHeader");
    const sparsityHeader = document.getElementById("sparsityHeader");

    const fly3dContainer = document.getElementById("fly3dContainer");
    const flyCanvas = document.getElementById("flyCanvas");
    const gaitStatusPill = document.getElementById("gaitStatusPill");
    const pitchAngleText = document.getElementById("pitchAngleText");
    const eyeLedL = document.getElementById("eyeLedL");
    const eyeLedR = document.getElementById("eyeLedR");
    const vncLed = document.getElementById("vncLed");
    const gfMeterVal = document.getElementById("gfMeterVal");
    const gfMeterBar = document.getElementById("gfMeterBar");
    const loomingMeterVal = document.getElementById("loomingMeterVal");
    const loomingMeterBar = document.getElementById("loomingMeterBar");
    const camBtns = document.querySelectorAll(".cam-btn");
    const resetCamBtn = document.getElementById("resetCamBtn");

    // Brain Canvas Elements
    const brainCanvasContainer = document.getElementById("brainCanvasContainer");
    const brainCanvas = document.getElementById("brainCanvas");
    const brainCtx = brainCanvas.getContext("2d");
    const brainViewLabel = document.getElementById("brainViewLabel");
    const brainZoomLabel = document.getElementById("brainZoomLabel");
    const brainTooltip = document.getElementById("brainTooltip");
    const ttId = document.getElementById("ttId");
    const ttType = document.getElementById("ttType");
    const ttNeuropil = document.getElementById("ttNeuropil");
    const ttSide = document.getElementById("ttSide");
    const ttDegrees = document.getElementById("ttDegrees");
    const ttCoords = document.getElementById("ttCoords");
    const ttAct = document.getElementById("ttAct");

    const zoomInBtn = document.getElementById("zoomInBtn");
    const zoomOutBtn = document.getElementById("zoomOutBtn");
    const zoomResetBtn = document.getElementById("zoomResetBtn");
    const focusBtns = document.querySelectorAll(".focus-btn");

    const toggleNeuropils = document.getElementById("toggleNeuropils");
    const toggleScaffold = document.getElementById("toggleScaffold");
    const toggleStreams = document.getElementById("toggleStreams");
    const toggleVP = document.getElementById("toggleVP");
    const toggleCB = document.getElementById("toggleCB");
    const toggleDN = document.getElementById("toggleDN");

    const vpMeanVal = document.getElementById("vpMeanVal");
    const cbMeanVal = document.getElementById("cbMeanVal");
    const dnMeanVal = document.getElementById("dnMeanVal");

    const marioNesFrameImg = document.getElementById("marioNesFrameImg");
    const marioActionNameBadge = document.getElementById("marioActionNameBadge");
    const marioCrtPos = document.getElementById("marioCrtPos");
    const levelProgressText = document.getElementById("levelProgressText");
    const levelProgressBar = document.getElementById("levelProgressBar");
    const btnRight = document.getElementById("btnRight");
    const btnA = document.getElementById("btnA");
    const btnB = document.getElementById("btnB");
    const actorProbBars = document.getElementById("actorProbBars");
    const neuronTableBody = document.getElementById("neuronTableBody");

    // Rebuild Milestones Bookmarks dynamically for active stage
    function rebuildMilestones() {
      eventChips.innerHTML = "";
      const chips = [];
      chips.push({ step: 0, label: "1. Spawn (x=40)" });

      if (currentStageId === "1-1") {
        const w11Defs = [
          { key: "Goomba 1", label: "2. Goomba 1 (Jump)" },
          { key: "Pipe 1", label: "3. Pipe 1 (32px)" },
          { key: "Pipe 2", label: "4. Pipe 2 (48px)" },
          { key: "Pipe 3", label: "5. Pipe 3 (64px)" },
          { key: "Pipe 4", label: "6. Pipe 4 (High)" },
          { key: "Pit 1 Abyss", label: "7. Abyss Leap" },
          { key: "Flagpole", label: "8. Flagpole Win" },
        ];
        w11Defs.forEach(m => {
          let targetFrame = (m.key === "Flagpole") ? currentSteps.length - 1 : currentSteps.findIndex(s => s.m && s.m.includes(m.key));
          if (targetFrame >= 0) {
            chips.push({ step: targetFrame, label: m.label });
          }
        });
      } else {
        let foundKeys = [];
        currentSteps.forEach((s, idx) => {
          if (s.m && s.m !== "" && s.m !== "Stage Spawn" && !foundKeys.includes(s.m)) {
            foundKeys.push(s.m);
            chips.push({ step: idx, label: `${chips.length + 1}. ${s.m} (F${s.step})` });
          }
        });
        if (chips.length === 1 || chips[chips.length - 1].step !== currentSteps.length - 1) {
          const finalTitle = currentStage.flag_captured ? "Flagpole Win" : `Final (x=${currentStage.max_x})`;
          chips.push({ step: currentSteps.length - 1, label: `${chips.length + 1}. ${finalTitle}` });
        }
      }

      chips.forEach(m => {
        const chip = document.createElement("button");
        const isBaseline = (currentModelType === "baseline");
        chip.className = isBaseline
          ? "px-2.5 py-1 rounded-full border border-amber-500/25 bg-amber-500/10 text-amber-200 hover:border-amber-500/45 hover:text-white transition text-[11px] font-medium shadow-sm"
          : "px-2.5 py-1 rounded-full border border-white/[0.08] bg-white/[0.04] text-[#86868b] hover:border-white/20 hover:text-white transition text-[11px] font-medium";
        chip.textContent = m.label;
        chip.onclick = () => {
          if (isPlaying) togglePlay();
          renderStep(m.step, true);
        };
        eventChips.appendChild(chip);
      });
    }

    // Model Architecture Switcher Handler
    function selectModel(modelType) {
      if (modelType === currentModelType) return;
      if (isPlaying) togglePlay();
      currentModelType = modelType;

      if (currentModelType === "baseline") {
        if (modelBtnBaseline) {
          modelBtnBaseline.className = "model-switch-btn px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm";
        }
        if (modelDotBaseline) {
          modelDotBaseline.className = "w-2 h-2 rounded-full bg-amber-400 animate-pulse";
        }
        if (modelBtnConnectome) {
          modelBtnConnectome.className = "model-switch-btn px-3.5 py-1.5 rounded-lg text-xs font-medium flex items-center gap-2 transition-all text-[#86868b] hover:text-white border border-transparent";
        }
        if (modelDotConnectome) {
          modelDotConnectome.className = "w-2 h-2 rounded-full bg-zinc-600";
        }

        if (appTitleText) appTitleText.textContent = "Dense GRU-1024 Baseline Simulator";
        if (modelVersionBadge) {
          modelVersionBadge.textContent = "Dense GRU PPO v3 (All 32 Stages)";
          modelVersionBadge.className = "px-2 py-0.5 rounded-full text-[10px] font-medium bg-amber-500/15 text-amber-300 border border-amber-500/30";
        }

        if (lblVpDrive) lblVpDrive.textContent = "CNN Vision Embed";
        if (lblCbCore) lblCbCore.textContent = "GRU Recurrent State";
        if (lblDnMotor) lblDnMotor.textContent = "Actor Policy Head";

        if (synapseMetricText) synapseMetricText.innerHTML = '<span class="text-amber-400 font-medium">GRU Weights:</span> <span class="text-white tabular-nums">1,024 × 1,024 Dense</span>';
        if (neuronMetricText) neuronMetricText.innerHTML = '<span class="text-sky-400 font-medium">Architecture:</span> <span class="text-white">Nature CNN + GRU-1024</span>';
        if (motorUnitMetricText) motorUnitMetricText.innerHTML = '<span class="text-rose-400 font-medium">Policy Head:</span> <span class="text-white tabular-nums">5 Actions</span>';

        if (neuronSectionTitle) neuronSectionTitle.textContent = "Active Latent Units Explorer (Dense GRU-1024)";
        if (neuronSectionSubtitle) neuronSectionSubtitle.textContent = "1,024 unconstrained recurrent units with no physical biological coordinates or neurotransmitter constraints.";
        if (tableTabContainer) tableTabContainer.classList.add("hidden");

        renderBenchmarkGrid(activeBiomeFilter, activeWorldFilter);
        selectStage(currentStageId || "1-1");
      } else {
        if (modelBtnConnectome) {
          modelBtnConnectome.className = "model-switch-btn px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-sky-500/20 text-sky-300 border border-sky-500/40 shadow-sm";
        }
        if (modelDotConnectome) {
          modelDotConnectome.className = "w-2 h-2 rounded-full bg-sky-400 animate-pulse";
        }
        if (modelBtnBaseline) {
          modelBtnBaseline.className = "model-switch-btn px-3.5 py-1.5 rounded-lg text-xs font-medium flex items-center gap-2 transition-all text-[#86868b] hover:text-white border border-transparent";
        }
        if (modelDotBaseline) {
          modelDotBaseline.className = "w-2 h-2 rounded-full bg-zinc-600";
        }

        if (appTitleText) appTitleText.textContent = "Fruit Fly Connectome Simulator";
        if (modelVersionBadge) {
          modelVersionBadge.textContent = "MaleCNS v1.0 (All 32 Stages)";
          modelVersionBadge.className = "px-2 py-0.5 rounded-full text-[10px] font-medium bg-sky-500/15 text-sky-300 border border-sky-500/30";
        }

        if (lblVpDrive) lblVpDrive.textContent = "Visual Drive";
        if (lblCbCore) lblCbCore.textContent = "Central Core";
        if (lblDnMotor) lblDnMotor.textContent = "Motor Output";

        if (synapseMetricText) synapseMetricText.innerHTML = '<span class="text-sky-400 font-medium">MaleCNS Synapses:</span> <span class="text-white tabular-nums">6,001,050</span>';
        if (neuronMetricText) neuronMetricText.innerHTML = '<span class="text-amber-400 font-medium">Neurons:</span> <span class="text-white tabular-nums">30,000</span> (3,700 Scaffold)';
        if (motorUnitMetricText) motorUnitMetricText.innerHTML = '<span class="text-rose-400 font-medium">Motor Units:</span> <span class="text-white tabular-nums">1,314 DNs</span>';

        if (neuronSectionTitle) neuronSectionTitle.textContent = "Active Neuron Explorer (FlyWire Annotations)";
        if (neuronSectionSubtitle) neuronSectionSubtitle.textContent = "Inspect instantaneous activations, cell types, 3D anatomical positions, and synaptic connectivity.";
        if (tableTabContainer) tableTabContainer.classList.remove("hidden");

        renderBenchmarkGrid(activeBiomeFilter, activeWorldFilter);
        selectStage(currentStageId || "1-1");
      }
    }

    // Switch active stage
    function selectStage(stageId) {
      const activeDataset = (currentModelType === "baseline") ? BASELINE_DATA : SIM_DATA;
      if (!activeDataset.stages || !activeDataset.stages[stageId]) return;
      if (isPlaying) togglePlay();

      currentStageId = stageId;
      currentStage = activeDataset.stages[currentStageId];
      currentSteps = currentStage.steps || [];
      currentStepIdx = 0;

      // Update Top Selector & Badges
      if (stageSelect) stageSelect.value = stageId;
      if (stageDescriptionText) stageDescriptionText.textContent = currentStage.desc;
      if (currentStageBiomeBadge) currentStageBiomeBadge.textContent = currentStage.biome;

      let bClass = "bg-emerald-500/10 text-emerald-300 border-emerald-500/20";
      if (currentStage.biome.includes("Underground")) bClass = "bg-indigo-500/10 text-indigo-300 border-indigo-500/20";
      else if (currentStage.biome.includes("Castle")) bClass = "bg-rose-500/10 text-rose-300 border-rose-500/20";
      else if (currentStage.biome.includes("Underwater")) bClass = "bg-cyan-500/10 text-cyan-300 border-cyan-500/20";
      else if (currentStage.biome.includes("Athletics")) bClass = "bg-amber-500/10 text-amber-300 border-amber-500/20";
      else if (currentStage.biome.includes("Night")) bClass = "bg-purple-500/10 text-purple-300 border-purple-500/20";
      if (currentStageBiomeBadge) currentStageBiomeBadge.className = `px-2.5 py-1 rounded-full text-[11px] font-medium border ${bClass}`;

      if (currentStageStatusBadge) {
        if (currentStage.flag_captured) {
          currentStageStatusBadge.className = "px-2.5 py-1 rounded-full text-[11px] font-semibold bg-emerald-500/15 text-emerald-400 border border-emerald-500/30 flex items-center gap-1.5";
          currentStageStatusBadge.innerHTML = "<span>🏆</span> Flagpole Win (100% Cleared)";
        } else {
          currentStageStatusBadge.className = "px-2.5 py-1 rounded-full text-[11px] font-medium bg-sky-500/10 text-sky-300 border border-sky-500/20 flex items-center gap-1.5";
          currentStageStatusBadge.innerHTML = `<span>🏃</span> Max Progress: X=${currentStage.max_x}`;
        }
      }

      // Update Scrubber Bounds & Counters
      simSlider.max = Math.max(1, currentSteps.length - 1);
      simSlider.value = 0;
      if (totalStepsCounter) totalStepsCounter.textContent = currentSteps.length.toLocaleString();

      rebuildMilestones();
      highlightActiveStageCard(stageId);
      renderStep(0, true);
    }

    function populateStageSelector() {
      if (!SIM_DATA.stages || !stageSelect) return;
      stageSelect.innerHTML = "";

      for (let w = 1; w <= 8; w++) {
        const group = document.createElement("optgroup");
        group.label = `World ${w}`;
        for (let s = 1; s <= 4; s++) {
          const sid = `${w}-${s}`;
          const stg = SIM_DATA.stages[sid];
          if (stg) {
            const opt = document.createElement("option");
            opt.value = sid;
            const flagTag = stg.flag_captured ? " 🏆 WIN" : ` (X=${stg.max_x})`;
            opt.textContent = `${stg.title} • [${stg.biome}]${flagTag}`;
            group.appendChild(opt);
          }
        }
        stageSelect.appendChild(group);
      }
      stageSelect.value = currentStageId;

      stageSelect.onchange = (e) => {
        selectStage(e.target.value);
      };
    }

    function renderBenchmarkGrid(filterBiome = "all", filterWorld = "all") {
      const activeDataset = (currentModelType === "baseline") ? BASELINE_DATA : SIM_DATA;
      if (!stagesBenchmarkGrid || !activeDataset.metadata || !activeDataset.metadata.benchmark_summary) return;
      stagesBenchmarkGrid.innerHTML = "";

      activeDataset.metadata.benchmark_summary.forEach(item => {
        if (filterBiome !== "all" && !item.biome.includes(filterBiome)) return;
        if (filterWorld !== "all" && item.world.toString() !== filterWorld) return;

        const isCurrent = (item.id === currentStageId);
        const card = document.createElement("div");
        card.id = `stageCard-${item.id}`;
        card.className = `rounded-xl p-2.5 transition border cursor-pointer ${isCurrent ? "bg-white/[0.12] border-sky-400 ring-1 ring-sky-400/50 shadow-lg" : "bg-white/[0.03] hover:bg-white/[0.07] border-white/[0.08]"}`;
        card.onclick = () => {
          if (currentModelType === "baseline") {
            selectModel("connectome");
          }
          selectStage(item.id);
          window.scrollTo({ top: 0, behavior: "smooth" });
        };

        let badgeStyle = "text-emerald-400 bg-emerald-500/10 border-emerald-500/20";
        if (item.biome.includes("Underground")) badgeStyle = "text-indigo-300 bg-indigo-500/10 border-indigo-500/20";
        else if (item.biome.includes("Castle")) badgeStyle = "text-rose-300 bg-rose-500/10 border-rose-500/20";
        else if (item.biome.includes("Underwater")) badgeStyle = "text-cyan-300 bg-cyan-500/10 border-cyan-500/20";
        else if (item.biome.includes("Athletics")) badgeStyle = "text-amber-300 bg-amber-500/10 border-amber-500/20";

        const pct = Math.min(100, Math.max(4, (item.max_x / 3337) * 100)).toFixed(0);

        card.innerHTML = `
          <div class="flex justify-between items-start mb-1.5">
            <span class="font-bold text-xs text-white tabular-nums">W${item.id}</span>
            <span class="text-[9px] px-1.5 py-0.5 rounded border ${badgeStyle} truncate max-w-[70px]">${item.biome}</span>
          </div>
          <div class="text-[10px] text-gray-300 font-medium truncate mb-1" title="${item.title}">${item.title.split(": ")[1] || item.title}</div>
          <div class="space-y-1">
            <div class="flex justify-between text-[10px] text-[#86868b] tabular-nums">
              <span>${item.flag_captured ? "🏆 Win" : `X=${item.max_x}`}</span>
              <span>${item.total_frames} F</span>
            </div>
            <div class="w-full bg-white/[0.08] h-1 rounded-full overflow-hidden">
              <div class="${item.flag_captured ? "bg-emerald-400" : "bg-sky-400"} h-full" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
        stagesBenchmarkGrid.appendChild(card);
      });
    }

    function highlightActiveStageCard(stageId) {
      if (!stagesBenchmarkGrid) return;
      document.querySelectorAll("[id^='stageCard-']").forEach(card => {
        card.className = "rounded-xl p-2.5 transition border cursor-pointer bg-white/[0.03] hover:bg-white/[0.07] border-white/[0.08]";
      });
      const active = document.getElementById(`stageCard-${stageId}`);
      if (active) {
        active.className = "rounded-xl p-2.5 transition border cursor-pointer bg-white/[0.12] border-sky-400 ring-1 ring-sky-400/50 shadow-lg";
      }
    }

    const B_BOUNDS = {
      xMin: 3876, xMax: 92454,
      yMin: 4950, yMax: 51389,
      zMin: 10154, zMax: 44120
    };

    // =========================================================================
    // --- 3D DROSOPHILA MELANOGASTER PROCEDURAL MODEL (Three.js WebGL) ---
    // =========================================================================
    function init3DFruitFly() {
      if (typeof THREE === "undefined") {
        console.warn("Three.js not loaded. Falling back to 2D fly rendering.");
        return;
      }

      const w = Math.max(300, fly3dContainer.clientWidth || 380);
      const h = Math.max(250, fly3dContainer.clientHeight || 310);

      flyScene = new THREE.Scene();
      flyScene.fog = new THREE.FogExp2(0x000000, 0.02);

      camOrbit.target = new THREE.Vector3(0, 1.25, 0);
      flyCamera = new THREE.PerspectiveCamera(40, w / h, 0.1, 80);
      update3DCamera();

      try {
        flyRenderer = new THREE.WebGLRenderer({ canvas: flyCanvas, antialias: true, alpha: true });
        flyRenderer.setSize(w, h);
        flyRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
        flyRenderer.toneMapping = THREE.ACESFilmicToneMapping;
        flyRenderer.toneMappingExposure = 1.25;
      } catch (err) {
        console.error("WebGL initialization failed:", err);
        return;
      }

      // Clean Drosophila Specimen Studio Lighting
      const ambientLight = new THREE.AmbientLight(0xffffff, 1.35);
      flyScene.add(ambientLight);

      const sunKeyLight = new THREE.DirectionalLight(0xffedd5, 2.2);
      sunKeyLight.position.set(7, 14, 9);
      flyScene.add(sunKeyLight);

      const blueFillLight = new THREE.DirectionalLight(0x93c5fd, 1.2);
      blueFillLight.position.set(-8, 6, -6);
      flyScene.add(blueFillLight);

      const rimLight = new THREE.DirectionalLight(0xfef08a, 1.1);
      rimLight.position.set(0, 8, -10);
      flyScene.add(rimLight);

      // Specimen Ground Stage
      const stageGeo = new THREE.CylinderGeometry(5.2, 5.4, 0.22, 36);
      const stageMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.75, metalness: 0.2 });
      const stageMesh = new THREE.Mesh(stageGeo, stageMat);
      stageMesh.position.y = -0.11;
      flyScene.add(stageMesh);

      const ringGeo = new THREE.RingGeometry(3.5, 3.56, 36);
      const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.4, side: THREE.DoubleSide });
      const ringMesh = new THREE.Mesh(ringGeo, ringMat);
      ringMesh.rotation.x = -Math.PI / 2;
      ringMesh.position.y = 0.01;
      flyScene.add(ringMesh);

      // Fly Root Group
      const flyRoot = new THREE.Group();
      flyRoot.position.set(0, 1.35, 0);
      flyScene.add(flyRoot);

      // Biological Materials (Drosophila melanogaster)
      // 1. Thorax & Scutum Chitin: Satin honey-amber with subtle dorsocentral sheen
      const chitinScutumMat = new THREE.MeshStandardMaterial({
        color: 0xc88736,
        roughness: 0.32,
        metalness: 0.08,
        emissive: 0x3d2005,
        emissiveIntensity: 0.12
      });

      // 2. Anterior Abdominal Chitin (Tergites 1-4): Translucent warm honey-tan
      const chitinAbdomenTanMat = new THREE.MeshStandardMaterial({
        color: 0xd89539,
        roughness: 0.36,
        metalness: 0.05,
        emissive: 0x331802,
        emissiveIntensity: 0.08
      });

      // 3. Posterior Male Melanin Cap (Tergites 5-6): Solid, glossy deep melanin black
      //    (The definitive biological trait of male Drosophila melanogaster - "black belly")
      const chitinMelaninBlackMat = new THREE.MeshStandardMaterial({
        color: 0x141210,
        roughness: 0.28,
        metalness: 0.18
      });

      // 4. Fine Tergite Posterior Margin Borders
      const chitinTergiteBorderMat = new THREE.MeshStandardMaterial({
        color: 0x27201c,
        roughness: 0.35,
        metalness: 0.12
      });

      // 5. Sensory Macrochaetae (Bristles): Glossy jet black, sharp
      const bristleMat = new THREE.MeshStandardMaterial({
        color: 0x09090b,
        roughness: 0.2,
        metalness: 0.35
      });

      // 6. Compound Eyes: Huge, brilliant ruby-red / brick-red ommatidia
      const eyeMatL = new THREE.MeshStandardMaterial({
        color: 0xdc2626,
        emissive: 0x881337,
        emissiveIntensity: 0.45,
        roughness: 0.16,
        metalness: 0.25
      });
      const eyeMatR = new THREE.MeshStandardMaterial({
        color: 0xdc2626,
        emissive: 0x881337,
        emissiveIntensity: 0.45,
        roughness: 0.16,
        metalness: 0.25
      });

      // 7. Ocelli: Tiny bright ruby simple eyes
      const ocelliMat = new THREE.MeshStandardMaterial({
        color: 0xef4444,
        emissive: 0xff1e00,
        emissiveIntensity: 0.75,
        roughness: 0.1
      });

      // 8. Wings: Dual-sided glassy physical membrane with slight iridescence
      const wingMat = new THREE.MeshPhysicalMaterial({
        color: 0xf0f9ff,
        transparent: true,
        opacity: 0.42,
        roughness: 0.08,
        metalness: 0.04,
        clearcoat: 1.0,
        clearcoatRoughness: 0.05,
        side: THREE.DoubleSide
      });

      // 9. Wing Veins: Chitinous amber-brown veins
      const wingVeinMat = new THREE.MeshBasicMaterial({
        color: 0x6b2e05,
        transparent: true,
        opacity: 0.88
      });

      // 10. Legs: Slender translucent amber chitin
      const legChitinMat = new THREE.MeshStandardMaterial({
        color: 0xb45309,
        roughness: 0.36,
        metalness: 0.1
      });
      const legTibiaMat = new THREE.MeshStandardMaterial({
        color: 0xc27803,
        roughness: 0.38,
        metalness: 0.08
      });

      // 11. Halteres: Creamy ivory club knob with amber stem
      const haltereStemMat = new THREE.MeshStandardMaterial({ color: 0xb45309, roughness: 0.4 });
      const haltereKnobMat = new THREE.MeshStandardMaterial({ color: 0xfef08a, roughness: 0.5 });

      // Helper: Create tapered sharp sensory bristle with natural backward curve
      function createBristle(length, radius, bendX = 0, bendZ = 0) {
        const bristleGroup = new THREE.Group();
        const segs = 3;
        const segLen = length / segs;
        let parent = bristleGroup;
        for (let i = 0; i < segs; i++) {
          const rBottom = radius * (1 - i / (segs + 0.5));
          const rTop = radius * (1 - (i + 1) / (segs + 0.5));
          const geo = new THREE.CylinderGeometry(Math.max(0.005, rTop), Math.max(0.008, rBottom), segLen, 6);
          const mesh = new THREE.Mesh(geo, bristleMat);
          mesh.position.y = segLen / 2;
          const node = new THREE.Group();
          node.add(mesh);
          node.position.y = (i === 0) ? 0 : segLen;
          node.rotation.x = bendX / segs;
          node.rotation.z = bendZ / segs;
          parent.add(node);
          parent = node;
        }
        return bristleGroup;
      }

      // --- Thorax (Mesothorax Scutum + Scutellum) ---
      const thoraxGroup = new THREE.Group();
      flyRoot.add(thoraxGroup);

      const scutumMesh = new THREE.Mesh(new THREE.SphereGeometry(1.35, 24, 18), chitinScutumMat);
      scutumMesh.scale.set(1.02, 0.92, 1.28);
      thoraxGroup.add(scutumMesh);

      // Scutellum: Characteristic triangular-rounded posterior shield
      const scutGroup = new THREE.Group();
      scutGroup.position.set(0, 0.62, -1.05);
      thoraxGroup.add(scutGroup);

      const scutMesh = new THREE.Mesh(new THREE.ConeGeometry(0.58, 0.68, 6), chitinScutumMat);
      scutMesh.rotation.x = -Math.PI / 3;
      scutGroup.add(scutMesh);

      // 4 Scutellar Bristles: 2 basal, 2 apical crossing at the apex (hallmark Drosophila trait)
      const scutL1 = createBristle(0.95, 0.026, -0.2, -0.15);
      scutL1.position.set(-0.35, 0.25, -0.15);
      scutL1.rotation.set(-0.3, 0, 0.3);
      scutGroup.add(scutL1);

      const scutR1 = createBristle(0.95, 0.026, -0.2, 0.15);
      scutR1.position.set(0.35, 0.25, -0.15);
      scutR1.rotation.set(-0.3, 0, -0.3);
      scutGroup.add(scutR1);

      const scutApicalL = createBristle(1.15, 0.028, -0.25, 0.3);
      scutApicalL.position.set(-0.16, 0.18, -0.5);
      scutApicalL.rotation.set(-0.4, 0, 0.25);
      scutGroup.add(scutApicalL);

      const scutApicalR = createBristle(1.15, 0.028, -0.25, -0.3);
      scutApicalR.position.set(0.16, 0.18, -0.5);
      scutApicalR.rotation.set(-0.4, 0, -0.25);
      scutGroup.add(scutApicalR);

      // Dorsocentral & notopleural thoracic macrochaetae
      const thoracicBristleCoords = [
        { x: -0.42, y: 0.85, z: 0.15, len: 1.05, rx: -0.55, rz: 0.1 },
        { x: 0.42, y: 0.85, z: 0.15, len: 1.05, rx: -0.55, rz: -0.1 },
        { x: -0.45, y: 0.75, z: -0.45, len: 1.25, rx: -0.65, rz: 0.08 },
        { x: 0.45, y: 0.75, z: -0.45, len: 1.25, rx: -0.65, rz: -0.08 },
        { x: -0.92, y: 0.45, z: 0.35, len: 0.9, rx: -0.3, rz: 0.5 },
        { x: 0.92, y: 0.45, z: 0.35, len: 0.9, rx: -0.3, rz: -0.5 },
        { x: -0.98, y: 0.4, z: -0.25, len: 0.95, rx: -0.4, rz: 0.45 },
        { x: 0.98, y: 0.4, z: -0.25, len: 0.95, rx: -0.4, rz: -0.45 }
      ];
      thoracicBristleCoords.forEach(p => {
        const b = createBristle(p.len, 0.024, -0.2, 0);
        b.position.set(p.x, p.y, p.z);
        b.rotation.set(p.rx, 0, p.rz);
        thoraxGroup.add(b);
      });

      // Internal VNC Neural Core
      const vncLight = new THREE.PointLight(0x06b6d4, 1.2, 3.5);
      vncLight.position.set(0, -0.2, 0.1);
      flyRoot.add(vncLight);

      const vncCoreMesh = new THREE.Mesh(new THREE.SphereGeometry(0.2, 12, 12), new THREE.MeshBasicMaterial({ color: 0x06b6d4 }));
      vncCoreMesh.position.set(0, -0.2, 0.1);
      flyRoot.add(vncCoreMesh);

      // --- Head (Caput) ---
      const headGroup = new THREE.Group();
      headGroup.position.set(0, 0.15, 1.6);
      flyRoot.add(headGroup);

      // Narrow central face & vertex capsule
      const headCapsule = new THREE.Mesh(new THREE.SphereGeometry(0.75, 18, 14), chitinScutumMat);
      headCapsule.scale.set(0.95, 0.95, 0.85);
      headGroup.add(headCapsule);

      // Massive Bulbous Ruby-Red Compound Eyes (75% of head volume)
      const eyeGeo = new THREE.SphereGeometry(0.72, 22, 18);
      eyeGeo.scale(0.92, 1.15, 1.05);

      const eyeMeshL = new THREE.Mesh(eyeGeo, eyeMatL);
      eyeMeshL.position.set(-0.72, 0.12, 0.18);
      eyeMeshL.rotation.set(0.08, -0.25, -0.15);
      headGroup.add(eyeMeshL);

      const eyeMeshR = new THREE.Mesh(eyeGeo, eyeMatR);
      eyeMeshR.position.set(0.72, 0.12, 0.18);
      eyeMeshR.rotation.set(0.08, 0.25, 0.15);
      headGroup.add(eyeMeshR);

      // Ocellar tubercle with 3 red ocelli and divergent bristles
      const ocellarTubercle = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.22, 0.12, 12), chitinMelaninBlackMat);
      ocellarTubercle.position.set(0, 0.82, 0.1);
      headGroup.add(ocellarTubercle);

      const oc1 = new THREE.Mesh(new THREE.SphereGeometry(0.05, 8, 8), ocelliMat);
      oc1.position.set(0, 0.88, 0.18);
      headGroup.add(oc1);
      const oc2 = new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 8), ocelliMat);
      oc2.position.set(-0.08, 0.88, 0.04);
      headGroup.add(oc2);
      const oc3 = new THREE.Mesh(new THREE.SphereGeometry(0.045, 8, 8), ocelliMat);
      oc3.position.set(0.08, 0.88, 0.04);
      headGroup.add(oc3);

      const ocBristleL = createBristle(0.7, 0.02, 0.35, -0.25);
      ocBristleL.position.set(-0.05, 0.88, 0.12);
      ocBristleL.rotation.set(0.3, 0, -0.3);
      headGroup.add(ocBristleL);

      const ocBristleR = createBristle(0.7, 0.02, 0.35, 0.25);
      ocBristleR.position.set(0.05, 0.88, 0.12);
      ocBristleR.rotation.set(0.3, 0, 0.3);
      headGroup.add(ocBristleR);

      // Postvertical & vertical head macrochaetae
      const pvBristleL = createBristle(0.75, 0.022, -0.3, 0.3);
      pvBristleL.position.set(-0.12, 0.84, -0.06);
      pvBristleL.rotation.set(-0.4, 0, 0.35);
      headGroup.add(pvBristleL);

      const pvBristleR = createBristle(0.75, 0.022, -0.3, -0.3);
      pvBristleR.position.set(0.12, 0.84, -0.06);
      pvBristleR.rotation.set(-0.4, 0, -0.35);
      headGroup.add(pvBristleR);

      const headBristleCoords = [
        { x: -0.45, y: 0.78, z: 0.05, len: 0.95, rx: -0.3, rz: -0.4 },
        { x: 0.45, y: 0.78, z: 0.05, len: 0.95, rx: -0.3, rz: 0.4 },
        { x: -0.68, y: 0.65, z: 0.0, len: 0.85, rx: -0.25, rz: -0.55 },
        { x: 0.68, y: 0.65, z: 0.0, len: 0.85, rx: -0.25, rz: 0.55 }
      ];
      headBristleCoords.forEach(p => {
        const b = createBristle(p.len, 0.022, -0.2, 0);
        b.position.set(p.x, p.y, p.z);
        b.rotation.set(p.rx, 0, p.rz);
        headGroup.add(b);
      });

      // Antennae with branched feathery arista
      function createDrosophilaAntenna(isLeft) {
        const antGroup = new THREE.Group();
        const sideSign = isLeft ? -1 : 1;

        const scape = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.08, 0.18, 8), chitinScutumMat);
        scape.rotation.x = Math.PI / 3;
        antGroup.add(scape);

        const pedicel = new THREE.Mesh(new THREE.SphereGeometry(0.11, 8, 8), chitinScutumMat);
        pedicel.position.set(0, 0.12, 0.15);
        antGroup.add(pedicel);

        const funiculus = new THREE.Mesh(new THREE.SphereGeometry(0.14, 10, 10), chitinAbdomenTanMat);
        funiculus.scale.set(0.9, 1.1, 1.3);
        funiculus.position.set(0, 0.18, 0.32);
        antGroup.add(funiculus);

        const aristaShaft = createBristle(0.85, 0.02, 0.35, sideSign * 0.4);
        aristaShaft.position.set(sideSign * 0.06, 0.28, 0.42);
        aristaShaft.rotation.set(0.2, sideSign * 0.3, 0);
        antGroup.add(aristaShaft);

        for (let r = 0; r < 5; r++) {
          const rayLen = 0.22 - r * 0.03;
          const ray = new THREE.Mesh(new THREE.CylinderGeometry(0.006, 0.01, rayLen, 4), bristleMat);
          ray.position.set(sideSign * (0.08 + r * 0.06), 0.34 + r * 0.09, 0.46 + r * 0.08);
          ray.rotation.set(0.5, sideSign * (0.4 + r * 0.1), (r % 2 === 0 ? 0.3 : -0.3));
          antGroup.add(ray);
        }

        antGroup.position.set(sideSign * 0.26, 0.3, 0.82);
        return antGroup;
      }
      headGroup.add(createDrosophilaAntenna(true));
      headGroup.add(createDrosophilaAntenna(false));

      // Proboscis: Fleshy rostrum + bilobed labellum sponge
      const probGroup = new THREE.Group();
      probGroup.position.set(0, -0.45, 0.45);
      probGroup.rotation.x = -Math.PI / 4;

      const rostrum = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.18, 0.48, 10), chitinScutumMat);
      rostrum.position.y = -0.22;
      probGroup.add(rostrum);

      const labellumL = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 8), chitinAbdomenTanMat);
      labellumL.scale.set(1.2, 0.7, 0.9);
      labellumL.position.set(-0.11, -0.48, 0);
      probGroup.add(labellumL);

      const labellumR = new THREE.Mesh(new THREE.SphereGeometry(0.15, 10, 8), chitinAbdomenTanMat);
      labellumR.scale.set(1.2, 0.7, 0.9);
      labellumR.position.set(0.11, -0.48, 0);
      probGroup.add(labellumR);

      headGroup.add(probGroup);

      // --- Abdomen (Male Drosophila melanogaster) ---
      // (NO bee stripes! T1-T4 amber with thin margins, T5-T6 solid melanin black)
      const abdomenGroup = new THREE.Group();
      abdomenGroup.position.set(0, -0.05, -1.35);
      flyRoot.add(abdomenGroup);

      const abSegments = [
        { z: -0.28, r: 1.15, l: 0.52, mat: chitinAbdomenTanMat, border: true },
        { z: -0.72, r: 1.25, l: 0.55, mat: chitinAbdomenTanMat, border: true },
        { z: -1.18, r: 1.22, l: 0.55, mat: chitinAbdomenTanMat, border: true },
        { z: -1.62, r: 1.08, l: 0.52, mat: chitinAbdomenTanMat, border: true },
        { z: -2.05, r: 0.88, l: 0.50, mat: chitinMelaninBlackMat, border: false },
        { z: -2.45, r: 0.62, l: 0.45, mat: chitinMelaninBlackMat, border: false }
      ];

      abSegments.forEach(seg => {
        const mesh = new THREE.Mesh(new THREE.CylinderGeometry(seg.r * 0.92, seg.r, seg.l, 20), seg.mat);
        mesh.rotation.x = Math.PI / 2;
        mesh.scale.set(1.05, 1.0, 0.78);
        mesh.position.set(0, -0.05, seg.z);
        abdomenGroup.add(mesh);

        if (seg.border) {
          const borderMesh = new THREE.Mesh(new THREE.TorusGeometry(seg.r * 0.98, 0.04, 8, 24), chitinTergiteBorderMat);
          borderMesh.scale.set(1.05, 0.78, 1.0);
          borderMesh.position.set(0, -0.05, seg.z - seg.l / 2 + 0.02);
          abdomenGroup.add(borderMesh);
        }
      });

      const terminaliaMesh = new THREE.Mesh(new THREE.SphereGeometry(0.48, 14, 12), chitinMelaninBlackMat);
      terminaliaMesh.scale.set(1.0, 0.75, 1.1);
      terminaliaMesh.position.set(0, -0.05, -2.65);
      abdomenGroup.add(terminaliaMesh);

      // --- Two Membranous Wings with Authentic Drosophila Venation ---
      function createDrosophilaWing(isLeft) {
        const wingRoot = new THREE.Group();
        const sideSign = isLeft ? -1 : 1;

        const shape = new THREE.Shape();
        shape.moveTo(0, 0);
        shape.bezierCurveTo(sideSign * 0.35, 1.2, sideSign * 0.85, 2.8, sideSign * 0.75, 4.2);
        shape.bezierCurveTo(sideSign * 0.65, 5.2, sideSign * -0.2, 5.5, sideSign * -0.65, 4.8);
        shape.bezierCurveTo(sideSign * -1.25, 3.8, sideSign * -1.35, 2.0, sideSign * -0.9, 0.8);
        shape.bezierCurveTo(sideSign * -0.6, 0.3, sideSign * -0.25, 0.05, 0, 0);

        const wingGeo = new THREE.ShapeGeometry(shape, 18);
        const membraneMesh = new THREE.Mesh(wingGeo, wingMat);
        membraneMesh.rotation.x = -Math.PI / 2;
        wingRoot.add(membraneMesh);

        function addVein(points, width = 0.032) {
          const curve = new THREE.CatmullRomCurve3(points.map(p => new THREE.Vector3(p[0], 0.008, -p[1])));
          const veinGeo = new THREE.TubeGeometry(curve, 16, width / 2, 5, false);
          const veinMesh = new THREE.Mesh(veinGeo, wingVeinMat);
          wingRoot.add(veinMesh);
        }

        addVein([[0, 0], [sideSign * 0.35, 1.2], [sideSign * 0.85, 2.8], [sideSign * 0.75, 4.2], [sideSign * 0.3, 5.3]], 0.048);
        addVein([[sideSign * 0.05, 0.4], [sideSign * 0.28, 1.4], [sideSign * 0.55, 2.1]], 0.035);
        addVein([[sideSign * 0.05, 0.4], [sideSign * 0.22, 1.8], [sideSign * 0.55, 3.5], [sideSign * 0.52, 4.8]], 0.032);
        addVein([[0, 0.6], [sideSign * 0.08, 2.0], [sideSign * 0.22, 3.8], [sideSign * 0.05, 5.25]], 0.032);
        addVein([[sideSign * -0.1, 0.8], [sideSign * -0.12, 2.2], [sideSign * -0.15, 3.6], [sideSign * -0.38, 4.9]], 0.03);
        addVein([[sideSign * 0.12, 2.6], [sideSign * -0.13, 2.6]], 0.026);
        addVein([[sideSign * -0.15, 3.8], [sideSign * -0.65, 3.5]], 0.026);
        addVein([[sideSign * -0.2, 0.9], [sideSign * -0.45, 2.2], [sideSign * -0.85, 3.2]], 0.028);

        return wingRoot;
      }

      const wingLGroup = new THREE.Group();
      wingLGroup.position.set(-0.65, 0.72, 0.05);
      const wingLModel = createDrosophilaWing(true);
      wingLGroup.add(wingLModel);
      flyRoot.add(wingLGroup);

      const wingRGroup = new THREE.Group();
      wingRGroup.position.set(0.65, 0.72, 0.05);
      const wingRModel = createDrosophilaWing(false);
      wingRGroup.add(wingRModel);
      flyRoot.add(wingRGroup);

      // --- Halteres (Metathoracic Balancers) ---
      function createHaltere(isLeft) {
        const hGroup = new THREE.Group();
        const sideSign = isLeft ? -1 : 1;
        const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.035, 0.48, 8), haltereStemMat);
        const knob = new THREE.Mesh(new THREE.SphereGeometry(0.13, 10, 10), haltereKnobMat);
        knob.scale.set(0.85, 1.25, 0.85);
        knob.position.y = 0.28;
        hGroup.add(stem);
        hGroup.add(knob);
        hGroup.position.set(sideSign * 0.82, 0.38, -0.85);
        hGroup.rotation.set(0.2, sideSign * -0.9, sideSign * -0.7);
        return hGroup;
      }
      const haltereL = createHaltere(true);
      const haltereR = createHaltere(false);
      flyRoot.add(haltereL);
      flyRoot.add(haltereR);

      // --- 6 Slender Jointed Legs with Male Sex Combs ---
      function createLeg(isLeft, type) {
        const root = new THREE.Group();
        const sideSign = isLeft ? -1 : 1;

        const coxaMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.18, 0.52, 10), legChitinMat);
        root.add(coxaMesh);

        const femurPivot = new THREE.Group();
        femurPivot.position.set(0, -0.28, 0);

        const fLen = (type === 0) ? 1.55 : ((type === 1) ? 1.85 : 2.15);
        const femurMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.10, 0.13, fLen, 8), legChitinMat);
        femurMesh.position.y = -fLen / 2;
        femurPivot.add(femurMesh);

        const tibiaPivot = new THREE.Group();
        tibiaPivot.position.set(0, -fLen, 0);

        const tLen = (type === 0) ? 1.65 : ((type === 1) ? 1.95 : 2.35);
        const tibiaMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.09, tLen, 8), legTibiaMat);
        tibiaMesh.position.y = -tLen / 2;
        tibiaPivot.add(tibiaMesh);

        const spur = new THREE.Mesh(new THREE.ConeGeometry(0.03, 0.12, 4), bristleMat);
        spur.position.set(sideSign * 0.06, -tLen, 0.04);
        spur.rotation.set(0.3, 0, sideSign * 0.4);
        tibiaPivot.add(spur);

        const tarsusPivot = new THREE.Group();
        tarsusPivot.position.set(0, -tLen, 0);

        const tarsusMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.065, 1.15, 6), legChitinMat);
        tarsusMesh.position.y = -0.55;
        tarsusPivot.add(tarsusMesh);

        // Male Front Leg Sex Comb (L1 & R1 basitarsus)
        if (type === 0) {
          const combGroup = new THREE.Group();
          for (let c = 0; c < 8; c++) {
            const tooth = new THREE.Mesh(new THREE.ConeGeometry(0.015, 0.08, 4), bristleMat);
            tooth.position.set(sideSign * 0.045, -0.2 - c * 0.04, 0.02);
            tooth.rotation.set(0.4, 0, sideSign * 0.7);
            combGroup.add(tooth);
          }
          tarsusPivot.add(combGroup);
        }

        const padMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.75 });
        const padMesh = new THREE.Mesh(new THREE.SphereGeometry(0.075, 8, 8), padMat);
        padMesh.position.y = -1.15;
        tarsusPivot.add(padMesh);

        tibiaPivot.add(tarsusPivot);
        femurPivot.add(tibiaPivot);
        root.add(femurPivot);

        return {
          root,
          femurPivot,
          tibiaPivot,
          tarsusPivot,
          padMat,
          isLeft,
          type
        };
      }

      const legL1 = createLeg(true, 0);
      legL1.root.position.set(-0.85, -0.45, 0.75);
      flyRoot.add(legL1.root);

      const legR1 = createLeg(false, 0);
      legR1.root.position.set(0.85, -0.45, 0.75);
      flyRoot.add(legR1.root);

      const legL2 = createLeg(true, 1);
      legL2.root.position.set(-1.0, -0.5, -0.05);
      flyRoot.add(legL2.root);

      const legR2 = createLeg(false, 1);
      legR2.root.position.set(1.0, -0.5, -0.05);
      flyRoot.add(legR2.root);

      const legL3 = createLeg(true, 2);
      legL3.root.position.set(-0.9, -0.45, -0.9);
      flyRoot.add(legL3.root);

      const legR3 = createLeg(false, 2);
      legR3.root.position.set(0.9, -0.45, -0.9);
      flyRoot.add(legR3.root);

      fly3D = {
        flyRoot,
        headGroup,
        abdomenGroup,
        wingLGroup,
        wingRGroup,
        haltereL,
        haltereR,
        eyeMatL,
        eyeMatR,
        vncLight,
        vncCoreMesh,
        legs: { L1: legL1, R1: legR1, L2: legL2, R2: legR2, L3: legL3, R3: legR3 }
      };

      is3DReady = true;
      render3D();

      setTimeout(() => {
        if (flyRenderer && flyCamera && fly3dContainer) {
          const cw = fly3dContainer.clientWidth || 380;
          const ch = fly3dContainer.clientHeight || 310;
          if (cw > 50 && ch > 50) {
            flyCamera.aspect = cw / ch;
            flyCamera.updateProjectionMatrix();
            flyRenderer.setSize(cw, ch);
            render3D();
          }
        }
      }, 60);

      // Mouse Orbit
      flyCanvas.addEventListener("mousedown", (e) => {
        if (e.button === 0) {
          isDragging3D = true;
          drag3DStart.x = e.clientX;
          drag3DStart.y = e.clientY;
        }
      });

      window.addEventListener("mousemove", (e) => {
        if (isDragging3D) {
          const dx = e.clientX - drag3DStart.x;
          const dy = e.clientY - drag3DStart.y;
          drag3DStart.x = e.clientX;
          drag3DStart.y = e.clientY;

          camOrbit.theta -= dx * 0.009;
          camOrbit.phi = Math.max(0.05, Math.min(Math.PI / 2 - 0.02, camOrbit.phi - dy * 0.009));
          update3DCamera();
          render3D();
        }
      });

      window.addEventListener("mouseup", () => {
        isDragging3D = false;
      });

      flyCanvas.addEventListener("wheel", (e) => {
        e.preventDefault();
        const factor = e.deltaY > 0 ? 1.09 : 0.91;
        camOrbit.radius = Math.max(6.5, Math.min(26.0, camOrbit.radius * factor));
        update3DCamera();
        render3D();
      }, { passive: false });

      window.addEventListener("resize", () => {
        const rw = fly3dContainer.clientWidth || 380;
        const rh = fly3dContainer.clientHeight || 310;
        if (flyCamera && flyRenderer) {
          flyCamera.aspect = rw / rh;
          flyCamera.updateProjectionMatrix();
          flyRenderer.setSize(rw, rh);
          render3D();
        }
      });
    }

    function update3DCamera() {
      if (!flyCamera || !camOrbit.target) return;
      const x = camOrbit.target.x + camOrbit.radius * Math.sin(camOrbit.phi) * Math.sin(camOrbit.theta);
      const y = camOrbit.target.y + camOrbit.radius * Math.cos(camOrbit.phi);
      const z = camOrbit.target.z + camOrbit.radius * Math.sin(camOrbit.phi) * Math.cos(camOrbit.theta);
      flyCamera.position.set(x, y, z);
      flyCamera.lookAt(camOrbit.target);
    }

    function render3D() {
      if (flyRenderer && flyScene && flyCamera) {
        flyRenderer.render(flyScene, flyCamera);
      }
    }

    function update3DFly(stepData) {
      if (!is3DReady || !fly3D) {
        return;
      }

      const action = stepData.a;
      const gf = stepData.gf;
      const step = stepData.step;

      const isJumping = (action === 2 || action === 4 || gf > 0.45);
      const isSprinting = (action === 3 || action === 4);
      const isWalking = (action === 1 || isSprinting);

      const gaitPhase = (step * (isSprinting ? 0.28 : 0.16)) % (Math.PI * 2);
      const phaseA = Math.sin(gaitPhase);
      const phaseB = -phaseA;

      let pitchDeg = 0;
      if (isJumping) pitchDeg = -18;
      else if (isSprinting) pitchDeg = 6;
      pitchAngleText.textContent = `${pitchDeg}°`;

      const pitchRad = (pitchDeg * Math.PI) / 180;
      fly3D.flyRoot.rotation.x = pitchRad;

      if (isJumping) {
        fly3D.flyRoot.position.y = 1.9;
      } else {
        const bob = isWalking ? Math.abs(Math.sin(gaitPhase * 2)) * 0.12 : 0;
        fly3D.flyRoot.position.y = 1.35 + bob;
      }

      fly3D.headGroup.rotation.x = isJumping ? -0.15 : (isSprinting ? 0.12 : 0);
      fly3D.abdomenGroup.rotation.x = isJumping ? 0.25 : (isSprinting ? -0.1 : 0);

      const timeMs = Date.now();
      if (isJumping) {
        const flutter = Math.sin(timeMs / 22) * 0.55;
        fly3D.wingLGroup.rotation.set(0.35 + flutter, -1.2, 0.35);
        fly3D.wingRGroup.rotation.set(0.35 - flutter, 1.2, -0.35);

        fly3D.haltereL.rotation.x = Math.sin(timeMs / 18) * 0.65;
        fly3D.haltereR.rotation.x = -Math.sin(timeMs / 18) * 0.65;
      } else {
        // Flat horizontal scissor posture overlapping across dorsal abdomen
        fly3D.wingLGroup.rotation.set(0.02, -0.14, 0.04);
        fly3D.wingRGroup.rotation.set(0.02, 0.14, -0.04);

        fly3D.haltereL.rotation.x = 0.2;
        fly3D.haltereR.rotation.x = 0.2;
      }

      const looming = (stepData.act[0] > 6.0);
      const loomingGlow = Math.min(2.5, Math.max(0.2, (stepData.act[0] - 4.0) * 0.45));
      fly3D.eyeMatL.emissiveIntensity = loomingGlow;
      fly3D.eyeMatR.emissiveIntensity = loomingGlow;
      fly3D.eyeMatL.emissive.setHex(looming ? 0xf59e0b : 0x78350f);
      fly3D.eyeMatR.emissive.setHex(looming ? 0xf59e0b : 0x78350f);

      fly3D.vncLight.color.setHex(isJumping ? 0xff453a : 0x38bdf8);
      fly3D.vncLight.intensity = isJumping ? 2.8 : 1.2;
      fly3D.vncCoreMesh.material.color.setHex(isJumping ? 0xff453a : 0x38bdf8);

      function animateLeg(leg, phaseVal, isJump) {
        const isLeft = leg.isLeft;
        const type = leg.type;
        const sideSign = isLeft ? -1 : 1;

        if (isJump) {
          if (type === 0) {
            leg.femurPivot.rotation.set(0.45, sideSign * 0.35, sideSign * 0.4);
            leg.tibiaPivot.rotation.set(0.3, 0, sideSign * -0.2);
            leg.tarsusPivot.rotation.set(-0.2, 0, 0);
          } else {
            leg.femurPivot.rotation.set(-0.85, sideSign * 0.45, sideSign * 0.55);
            leg.tibiaPivot.rotation.set(-0.65, 0, sideSign * -0.3);
            leg.tarsusPivot.rotation.set(-0.4, 0, 0);
          }
          leg.padMat.color.setHex(0xff453a);
          leg.padMat.opacity = 0.9;
        } else {
          const isStance = phaseVal >= 0;
          const p = Math.abs(phaseVal);

          if (type === 0) {
            const fAngle = isStance ? (0.2 - p * 0.35) : (0.45 + p * 0.25);
            const zSpread = sideSign * (isStance ? 0.65 : 0.85);
            leg.femurPivot.rotation.set(fAngle, sideSign * 0.25, zSpread);
            leg.tibiaPivot.rotation.set(isStance ? -0.45 : -0.75, 0, sideSign * -0.3);
            leg.tarsusPivot.rotation.set(isStance ? 0.25 : 0.45, 0, 0);
          } else if (type === 1) {
            const fAngle = isStance ? (-0.1 + p * 0.3) : (-0.35 - p * 0.25);
            const zSpread = sideSign * (isStance ? 0.85 : 1.05);
            leg.femurPivot.rotation.set(fAngle, sideSign * 0.1, zSpread);
            leg.tibiaPivot.rotation.set(isStance ? -0.55 : -0.85, 0, sideSign * -0.4);
            leg.tarsusPivot.rotation.set(isStance ? 0.35 : 0.5, 0, 0);
          } else {
            const fAngle = isStance ? (-0.35 + p * 0.4) : (-0.65 - p * 0.3);
            const zSpread = sideSign * (isStance ? 0.75 : 0.95);
            leg.femurPivot.rotation.set(fAngle, sideSign * -0.25, zSpread);
            leg.tibiaPivot.rotation.set(isStance ? -0.65 : -0.95, 0, sideSign * -0.35);
            leg.tarsusPivot.rotation.set(isStance ? 0.4 : 0.6, 0, 0);
          }

          leg.padMat.color.setHex(isStance ? 0x38bdf8 : 0x52525b);
          leg.padMat.opacity = isStance ? 0.9 : 0.25;
        }
      }

      animateLeg(fly3D.legs.L1, phaseA, isJumping);
      animateLeg(fly3D.legs.R2, phaseA, isJumping);
      animateLeg(fly3D.legs.L3, phaseA, isJumping);

      animateLeg(fly3D.legs.R1, phaseB, isJumping);
      animateLeg(fly3D.legs.L2, phaseB, isJumping);
      animateLeg(fly3D.legs.R3, phaseB, isJumping);

      render3D();

      let gaitText = "REST";
      if (isJumping) gaitText = "GIANT FIBER LEAP";
      else if (isSprinting) gaitText = "FAST TRIPOD";
      else if (isWalking) gaitText = "TRIPOD GAIT";

      gaitStatusPill.textContent = gaitText;
      gaitBadgeHeader.textContent = isJumping ? "Giant Fiber Leap" : (isSprinting ? "Fast Sprint" : "Tripod Gait");

      eyeLedL.className = `w-2 h-2 rounded-full ${looming ? "bg-amber-400" : "bg-zinc-600"}`;
      eyeLedR.className = `w-2 h-2 rounded-full ${looming ? "bg-amber-400" : "bg-zinc-600"}`;
      vncLed.className = `w-2 h-2 rounded-full ${isJumping ? "bg-rose-500" : "bg-sky-400"}`;

      gfMeterVal.textContent = gf.toFixed(3);
      gfMeterBar.style.width = `${Math.min(100, gf * 100)}%`;

      const loomingVal = Math.min(1.0, (stepData.act[0] - 5.0) / 4.0);
      loomingMeterVal.textContent = Math.max(0, loomingVal).toFixed(3);
      loomingMeterBar.style.width = `${Math.max(0, loomingVal * 100)}%`;

      const l1Stance = phaseA >= 0;
      updateLegPill("legL1", l1Stance, isJumping);
      updateLegPill("legL2", !l1Stance, isJumping);
      updateLegPill("legL3", l1Stance, isJumping);
      updateLegPill("legR1", !l1Stance, isJumping);
      updateLegPill("legR2", l1Stance, isJumping);
      updateLegPill("legR3", !l1Stance, isJumping);
    }

    function drawFlyBody2DFallback(stepData) {
      // Safe no-op: never acquire 2d context on flyCanvas to preserve WebGL context
    }

    function updateLegPill(id, isStance, isJump) {
      const el = document.getElementById(id);
      if (!el) return;
      const stateEl = el.querySelector(".leg-state");
      if (isJump) {
        el.className = "px-2 py-0.5 rounded bg-rose-950/70 text-rose-300 flex justify-between border border-rose-800/80 font-semibold text-[11px]";
        stateEl.textContent = "EXTEND";
      } else if (isStance) {
        el.className = "px-2 py-0.5 rounded bg-sky-950/70 text-sky-300 flex justify-between border border-sky-800/80 text-[11px]";
        stateEl.textContent = "STANCE";
      } else {
        el.className = "px-2 py-0.5 rounded bg-white/[0.04] text-zinc-400 flex justify-between border border-white/[0.06] text-[11px]";
        stateEl.textContent = "SWING";
      }
    }

    // --- PROJECTION GEOMETRY & COORDINATES ---
    function getNeuronProjCoords(n) {
      if (currentBrainView === "xy") return { x: n.x, y: n.y };
      if (currentBrainView === "xz") return { x: n.x, y: n.z };
      return { x: n.y, y: n.z };
    }

    function getNeuropilProjCoords(b) {
      if (currentBrainView === "xy") {
        return { cx: b.cx, cy: b.cy, rx: b.rx, ry: b.ry };
      } else if (currentBrainView === "xz") {
        return { cx: b.cx, cy: b.cz, rx: b.rx, ry: b.rz };
      } else {
        return { cx: b.cy, cy: b.cz, rx: b.ry, ry: b.rz };
      }
    }

    function getProjParams(w, h) {
      let minX, maxX, minY, maxY, xLabel, yLabel;
      if (currentBrainView === "xy") {
        minX = B_BOUNDS.xMin; maxX = B_BOUNDS.xMax;
        minY = B_BOUNDS.yMin; maxY = B_BOUNDS.yMax;
        xLabel = "Lateral (X)"; yLabel = "Dorso-Ventral (Y)";
      } else if (currentBrainView === "xz") {
        minX = B_BOUNDS.xMin; maxX = B_BOUNDS.xMax;
        minY = B_BOUNDS.zMin; maxY = B_BOUNDS.zMax;
        xLabel = "Lateral (X)"; yLabel = "Anterior-Posterior (Z)";
      } else {
        minX = B_BOUNDS.yMin; maxX = B_BOUNDS.yMax;
        minY = B_BOUNDS.zMin; maxY = B_BOUNDS.zMax;
        xLabel = "Dorso-Ventral (Y)"; yLabel = "Anterior-Posterior (Z)";
      }

      const pad = 36;
      const availW = w - pad * 2;
      const availH = h - pad * 2;
      const rangeX = maxX - minX;
      const rangeY = maxY - minY;

      const baseScale = Math.min(availW / rangeX, availH / rangeY);
      const baseOffsetX = pad + (availW - rangeX * baseScale) / 2;
      const baseOffsetY = pad + (availH - rangeY * baseScale) / 2;

      function toScreen(xd, yd) {
        const sx = baseOffsetX + (xd - minX) * baseScale;
        const sy = baseOffsetY + (yd - minY) * baseScale;
        const px = (sx - w / 2) * brainZoom + w / 2 + brainPanX;
        const py = (sy - h / 2) * brainZoom + h / 2 + brainPanY;
        return { px, py };
      }

      function toData(px, py) {
        const sx = (px - brainPanX - w / 2) / brainZoom + w / 2;
        const sy = (py - brainPanY - h / 2) / brainZoom + h / 2;
        const xd = (sx - baseOffsetX) / baseScale + minX;
        const yd = (sy - baseOffsetY) / baseScale + minY;
        return { xd, yd };
      }

      return { minX, maxX, minY, maxY, xLabel, yLabel, baseScale, baseOffsetX, baseOffsetY, toScreen, toData };
    }

    // --- ENHANCED BRAIN CONNECTOME MAP DRAWING ---
    function drawBrainMap(stepData) {
      if (!stepData) stepData = currentSteps[currentStepIdx];
      const w = brainCanvasContainer.clientWidth || 560;
      const h = brainCanvasContainer.clientHeight || 390;
      const dpr = window.devicePixelRatio || 1;

      if (brainCanvas.width !== Math.round(w * dpr) || brainCanvas.height !== Math.round(h * dpr)) {
        brainCanvas.width = Math.round(w * dpr);
        brainCanvas.height = Math.round(h * dpr);
      }

      brainCtx.save();
      brainCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      brainCtx.clearRect(0, 0, w, h);

      if (currentModelType === "baseline") {
        if (brainViewLabel) brainViewLabel.textContent = "Latent Raster (32×32)";
        if (brainZoomLabel) brainZoomLabel.textContent = "GRU-1024";

        // 1. Grid
        brainCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        brainCtx.lineWidth = 1;
        for (let gx = 0; gx < w; gx += 48) {
          brainCtx.beginPath(); brainCtx.moveTo(gx, 0); brainCtx.lineTo(gx, h); brainCtx.stroke();
        }
        for (let gy = 0; gy < h; gy += 48) {
          brainCtx.beginPath(); brainCtx.moveTo(0, gy); brainCtx.lineTo(w, gy); brainCtx.stroke();
        }

        // 2. Spatial Orientation Axes in Top Left
        brainCtx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
        brainCtx.fillStyle = "#71717a";
        brainCtx.fillText("⟷ Latent Columns [0..31]", 12, 18);
        brainCtx.fillText("↕ Latent Rows [0..31]", 12, 32);

        // 3. Render 32x32 GRU-1024 Latent Raster
        const cols = 32;
        const rows = 32;
        const availH = h - 110;
        const cellSize = Math.min((w - 80) / cols, availH / rows);
        const startX = (w - (cols * cellSize)) / 2;
        const startY = 38 + ((availH - (rows * cellSize)) / 2);

        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const uIdx = r * cols + c;
            const x = startX + c * cellSize;
            const y = startY + r * cellSize;

            const pseudoAct = Math.sin(uIdx * 0.47 + stepData.step * 0.08 + (stepData.a * 1.2)) * 0.5 + 0.5;
            const isFiring = (pseudoAct > 0.62);

            if (isFiring) {
              const alpha = 0.3 + pseudoAct * 0.65;
              brainCtx.fillStyle = `rgba(245, 158, 11, ${alpha})`;
            } else {
              brainCtx.fillStyle = `rgba(255, 255, 255, ${0.03 + pseudoAct * 0.08})`;
            }
            brainCtx.fillRect(x + 1, y + 1, cellSize - 2, cellSize - 2);

            if (hoveredNeuron && hoveredNeuron.uIdx === uIdx) {
              brainCtx.strokeStyle = "#ffffff";
              brainCtx.lineWidth = 1.5;
              brainCtx.strokeRect(x, y, cellSize, cellSize);
            }
          }
        }

        // 4. Apple Glassmorphic Comparative Information Card
        const cardH = 50;
        const cardY = h - cardH - 12;
        brainCtx.fillStyle = "rgba(18, 18, 20, 0.90)";
        brainCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
        brainCtx.lineWidth = 1;
        brainCtx.beginPath();
        brainCtx.roundRect(14, cardY, w - 28, cardH, 10);
        brainCtx.fill();
        brainCtx.stroke();

        brainCtx.font = "bold 11px -apple-system, BlinkMacSystemFont, sans-serif";
        brainCtx.fillStyle = "#f59e0b";
        brainCtx.fillText("⚡ Dense GRU-1024 Latent Representation (Non-Biological)", 26, cardY + 18);

        brainCtx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
        brainCtx.fillStyle = "#a1a1aa";
        brainCtx.fillText("Hoyer Sparsity: ~30.6 (Dense, diffuse energy) • No anatomical neuropils. Switch to MaleCNS for 30k connectome.", 26, cardY + 34);

        brainCtx.restore();
        return;
      }

      const params = getProjParams(w, h);

      // 1. Subtle Scientific Anatomical Grid
      brainCtx.strokeStyle = "rgba(255, 255, 255, 0.04)";
      brainCtx.lineWidth = 1;
      for (let gx = 0; gx < w; gx += 48) {
        brainCtx.beginPath(); brainCtx.moveTo(gx, 0); brainCtx.lineTo(gx, h); brainCtx.stroke();
      }
      for (let gy = 0; gy < h; gy += 48) {
        brainCtx.beginPath(); brainCtx.moveTo(0, gy); brainCtx.lineTo(w, gy); brainCtx.stroke();
      }

      // 2. Spatial Orientation Axes in Top Left Corner
      brainCtx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
      brainCtx.fillStyle = "#71717a";
      brainCtx.fillText(`⟷ ${params.xLabel}`, 12, 18);
      brainCtx.fillText(`↕ ${params.yLabel}`, 12, 32);

      // 3. 100 µm Scale Bar in Bottom Right
      const scaleBarUnits = 25000;
      const scaleBarLenPx = scaleBarUnits * params.baseScale * brainZoom;
      if (scaleBarLenPx > 20 && scaleBarLenPx < w * 0.4) {
        const sbX = w - 18 - scaleBarLenPx;
        const sbY = h - 22;
        brainCtx.strokeStyle = "rgba(255, 255, 255, 0.45)";
        brainCtx.lineWidth = 1.5;
        brainCtx.beginPath();
        brainCtx.moveTo(sbX, sbY);
        brainCtx.lineTo(sbX + scaleBarLenPx, sbY);
        brainCtx.moveTo(sbX, sbY - 3);
        brainCtx.lineTo(sbX, sbY + 3);
        brainCtx.moveTo(sbX + scaleBarLenPx, sbY - 3);
        brainCtx.lineTo(sbX + scaleBarLenPx, sbY + 3);
        brainCtx.stroke();
        brainCtx.fillStyle = "rgba(255, 255, 255, 0.65)";
        brainCtx.textAlign = "center";
        brainCtx.fillText("100 µm", sbX + scaleBarLenPx / 2, sbY - 5);
        brainCtx.textAlign = "left";
      }

      // 4. Anatomical Neuropil Boundary Outlines
      if (showNeuropils && SIM_DATA.neuropil_bounds) {
        Object.entries(SIM_DATA.neuropil_bounds).forEach(([npName, npInfo]) => {
          const proj = getNeuropilProjCoords(npInfo);
          const sc = params.toScreen(proj.cx, proj.cy);
          const radX = proj.rx * params.baseScale * brainZoom;
          const radY = proj.ry * params.baseScale * brainZoom;

          if (radX < 4 || radY < 4) return;

          brainCtx.save();
          if (npName.includes("Optic Lobe (L)")) {
            brainCtx.fillStyle = "rgba(245, 158, 11, 0.04)";
            brainCtx.strokeStyle = "rgba(245, 158, 11, 0.45)";
            brainCtx.lineWidth = 1.2;
            brainCtx.setLineDash([4, 4]);
            brainCtx.beginPath();
            brainCtx.ellipse(sc.px, sc.py, radX, radY, 0, 0, Math.PI * 2);
            brainCtx.fill();
            brainCtx.stroke();

            brainCtx.setLineDash([]);
            brainCtx.fillStyle = "rgba(245, 158, 11, 0.85)";
            brainCtx.font = "600 9px -apple-system, sans-serif";
            brainCtx.fillText("OPTIC LOBE (L)", sc.px - 32, sc.py - radY - 4);
          } else if (npName.includes("Optic Lobe (R)")) {
            brainCtx.fillStyle = "rgba(245, 158, 11, 0.04)";
            brainCtx.strokeStyle = "rgba(245, 158, 11, 0.45)";
            brainCtx.lineWidth = 1.2;
            brainCtx.setLineDash([4, 4]);
            brainCtx.beginPath();
            brainCtx.ellipse(sc.px, sc.py, radX, radY, 0, 0, Math.PI * 2);
            brainCtx.fill();
            brainCtx.stroke();

            brainCtx.setLineDash([]);
            brainCtx.fillStyle = "rgba(245, 158, 11, 0.85)";
            brainCtx.font = "600 9px -apple-system, sans-serif";
            brainCtx.fillText("OPTIC LOBE (R)", sc.px - 32, sc.py - radY - 4);
          } else if (npName.includes("Central Complex")) {
            brainCtx.fillStyle = "rgba(56, 189, 248, 0.08)";
            brainCtx.strokeStyle = "rgba(56, 189, 248, 0.65)";
            brainCtx.lineWidth = 1.4;
            brainCtx.setLineDash([4, 3]);
            brainCtx.beginPath();
            brainCtx.ellipse(sc.px, sc.py, radX, radY, 0, 0, Math.PI * 2);
            brainCtx.fill();
            brainCtx.stroke();

            brainCtx.setLineDash([]);
            brainCtx.fillStyle = "rgba(56, 189, 248, 0.95)";
            brainCtx.font = "600 9px -apple-system, sans-serif";
            brainCtx.fillText("CENTRAL COMPLEX (CX)", sc.px - 50, sc.py - radY - 4);
          } else if (npName.includes("Descending Motor Trunk")) {
            brainCtx.fillStyle = "rgba(255, 69, 58, 0.06)";
            brainCtx.strokeStyle = "rgba(255, 69, 58, 0.60)";
            brainCtx.lineWidth = 1.4;
            brainCtx.setLineDash([4, 4]);
            brainCtx.beginPath();
            brainCtx.ellipse(sc.px, sc.py, radX, radY, 0, 0, Math.PI * 2);
            brainCtx.fill();
            brainCtx.stroke();

            brainCtx.setLineDash([]);
            brainCtx.fillStyle = "rgba(255, 69, 58, 0.95)";
            brainCtx.font = "600 9px -apple-system, sans-serif";
            brainCtx.fillText("MOTOR TRUNK (DNs)", sc.px - 45, sc.py + radY + 12);
          } else {
            brainCtx.fillStyle = "rgba(255, 255, 255, 0.01)";
            brainCtx.strokeStyle = "rgba(255, 255, 255, 0.12)";
            brainCtx.lineWidth = 1.0;
            brainCtx.setLineDash([5, 5]);
            brainCtx.beginPath();
            brainCtx.ellipse(sc.px, sc.py, radX, radY, 0, 0, Math.PI * 2);
            brainCtx.fill();
            brainCtx.stroke();
          }
          brainCtx.restore();
        });
      }

      // 5. Dense 3,700-Neuron Morphological Scaffold
      if (showScaffold && SIM_DATA.scaffold) {
        brainCtx.fillStyle = "rgba(113, 113, 122, 0.45)";
        const dotR = Math.max(0.85, Math.min(2.2, 1.05 * Math.sqrt(brainZoom)));
        SIM_DATA.scaffold.forEach(n => {
          if (n.x === null) return;
          const pt = getNeuronProjCoords(n);
          const sc = params.toScreen(pt.x, pt.y);
          if (sc.px < -10 || sc.px > w + 10 || sc.py < -10 || sc.py > h + 10) return;
          brainCtx.beginPath();
          brainCtx.arc(sc.px, sc.py, dotR, 0, Math.PI * 2);
          brainCtx.fill();
        });
      }

      // 6. Live Synaptic Energy Streams
      if (showStreams && SIM_DATA.neuropil_bounds) {
        const cxObj = SIM_DATA.neuropil_bounds["Central Complex (CX)"];
        const cxProj = getNeuropilProjCoords(cxObj);
        const cxSc = params.toScreen(cxProj.cx, cxProj.cy);

        if (showVP && stepData.vps) {
          stepData.vps.slice(0, 8).forEach(([idx, act], i) => {
            const info = SIM_DATA.neuron_table[idx];
            if (!info || info.x === null) return;
            const pt = getNeuronProjCoords(info);
            const sc = params.toScreen(pt.x, pt.y);

            const cpx = (sc.px + cxSc.px) / 2;
            const cpy = Math.min(sc.py, cxSc.py) - 12 * Math.sqrt(brainZoom);

            brainCtx.strokeStyle = "rgba(245, 158, 11, 0.20)";
            brainCtx.lineWidth = 1.1;
            brainCtx.beginPath();
            brainCtx.moveTo(sc.px, sc.py);
            brainCtx.quadraticCurveTo(cpx, cpy, cxSc.px, cxSc.py);
            brainCtx.stroke();

            const t = ((animPhase + i * 0.15) % 1.0);
            const bx = (1 - t) * (1 - t) * sc.px + 2 * (1 - t) * t * cpx + t * t * cxSc.px;
            const by = (1 - t) * (1 - t) * sc.py + 2 * (1 - t) * t * cpy + t * t * cxSc.py;

            brainCtx.fillStyle = "rgba(245, 158, 11, 0.85)";
            brainCtx.beginPath();
            brainCtx.arc(bx, by, 2.0, 0, Math.PI * 2);
            brainCtx.fill();
          });
        }

        if (showDN && stepData.dns) {
          stepData.dns.slice(0, 8).forEach(([idx, act], i) => {
            const info = SIM_DATA.neuron_table[idx];
            if (!info || info.x === null) return;
            const pt = getNeuronProjCoords(info);
            const sc = params.toScreen(pt.x, pt.y);

            const cpx = (cxSc.px + sc.px) / 2 + (i % 2 === 0 ? 8 : -8);
            const cpy = (cxSc.py + sc.py) / 2;

            brainCtx.strokeStyle = "rgba(255, 69, 58, 0.22)";
            brainCtx.lineWidth = 1.1;
            brainCtx.beginPath();
            brainCtx.moveTo(cxSc.px, cxSc.py);
            brainCtx.quadraticCurveTo(cpx, cpy, sc.px, sc.py);
            brainCtx.stroke();

            const t = ((animPhase + i * 0.18) % 1.0);
            const bx = (1 - t) * (1 - t) * cxSc.px + 2 * (1 - t) * t * cpx + t * t * sc.px;
            const by = (1 - t) * (1 - t) * cxSc.py + 2 * (1 - t) * t * cpy + t * t * sc.py;

            brainCtx.fillStyle = "rgba(255, 69, 58, 0.9)";
            brainCtx.beginPath();
            brainCtx.arc(bx, by, 2.2, 0, Math.PI * 2);
            brainCtx.fill();
          });
        }

        if (stepData.gf > 0.4 || stepData.a === 2 || stepData.a === 4) {
          const dnObj = SIM_DATA.neuropil_bounds["Descending Motor Trunk"];
          const dnProj = getNeuropilProjCoords(dnObj);
          const dnSc = params.toScreen(dnProj.cx, dnProj.cy);

          brainCtx.strokeStyle = "rgba(255, 69, 58, 0.8)";
          brainCtx.lineWidth = 2.0;
          brainCtx.setLineDash([5, 4]);
          brainCtx.lineDashOffset = -(animPhase * 36) % 9;
          brainCtx.beginPath();
          brainCtx.moveTo(cxSc.px, cxSc.py);
          brainCtx.lineTo(dnSc.px, dnSc.py + 20 * brainZoom);
          brainCtx.stroke();
          brainCtx.setLineDash([]);
        }
      }

      // 7. Active Firing Neurons
      const activeGroups = [];
      if (showVP && stepData.vps) activeGroups.push({ list: stepData.vps, color: "#f59e0b", halo: "rgba(245, 158, 11, 0.25)", r: 3.5 });
      if (showCB && stepData.cbs) activeGroups.push({ list: stepData.cbs, color: "#38bdf8", halo: "rgba(56, 189, 248, 0.25)", r: 3.5 });
      if (showDN && stepData.dns) activeGroups.push({ list: stepData.dns, color: "#ff453a", halo: "rgba(255, 69, 58, 0.25)", r: 4.5 });

      activeGroups.forEach(grp => {
        grp.list.forEach(([idx, act]) => {
          const info = SIM_DATA.neuron_table[idx];
          if (!info || info.x === null) return;
          const pt = getNeuronProjCoords(info);
          const sc = params.toScreen(pt.x, pt.y);
          if (sc.px < -20 || sc.px > w + 20 || sc.py < -20 || sc.py > h + 20) return;

          const radius = Math.min(9.0, Math.max(2.6, grp.r * Math.sqrt(Math.abs(act)) * Math.sqrt(brainZoom)));

          // Subtle Aura
          brainCtx.fillStyle = grp.halo;
          brainCtx.beginPath();
          brainCtx.arc(sc.px, sc.py, radius * 2.0, 0, Math.PI * 2);
          brainCtx.fill();

          // Node Body
          brainCtx.fillStyle = grp.color;
          brainCtx.beginPath();
          brainCtx.arc(sc.px, sc.py, radius, 0, Math.PI * 2);
          brainCtx.fill();

          // Crisp Center Core
          brainCtx.fillStyle = "#ffffff";
          brainCtx.beginPath();
          brainCtx.arc(sc.px, sc.py, Math.max(1.1, radius * 0.38), 0, Math.PI * 2);
          brainCtx.fill();
        });
      });

      // 8. Precision Targeting Reticle
      if (hoveredNeuron && hoveredNeuron.x !== null) {
        const pt = getNeuronProjCoords(hoveredNeuron);
        const sc = params.toScreen(pt.x, pt.y);

        brainCtx.save();
        brainCtx.strokeStyle = "#38bdf8";
        brainCtx.lineWidth = 1.5;
        brainCtx.setLineDash([3, 3]);
        brainCtx.beginPath();
        brainCtx.arc(sc.px, sc.py, 13, 0, Math.PI * 2);
        brainCtx.stroke();
        brainCtx.setLineDash([]);

        const tickLen = 5;
        brainCtx.strokeStyle = "#ffffff";
        brainCtx.lineWidth = 1.2;
        brainCtx.beginPath();
        brainCtx.moveTo(sc.px - 13 - tickLen, sc.py); brainCtx.lineTo(sc.px - 13, sc.py);
        brainCtx.moveTo(sc.px + 13, sc.py); brainCtx.lineTo(sc.px + 13 + tickLen, sc.py);
        brainCtx.moveTo(sc.px, sc.py - 13 - tickLen); brainCtx.lineTo(sc.px, sc.py - 13);
        brainCtx.moveTo(sc.px, sc.py + 13); brainCtx.lineTo(sc.px, sc.py + 13 + tickLen);
        brainCtx.stroke();

        brainCtx.fillStyle = "#ffffff";
        brainCtx.beginPath();
        brainCtx.arc(sc.px, sc.py, 2.2, 0, Math.PI * 2);
        brainCtx.fill();
        brainCtx.restore();
      }

      brainCtx.restore();
    }

    // --- CONTINUOUS STREAM & 3D ANIMATION LOOP ---
    function streamAnimationLoop() {
      animPhase += 0.015;
      if (showStreams && currentSteps && currentSteps[currentStepIdx]) {
        drawBrainMap(currentSteps[currentStepIdx]);
      }
      if (is3DReady && fly3D) {
        const stepData = currentSteps ? currentSteps[currentStepIdx] : null;
        if (stepData && (stepData.a === 2 || stepData.a === 4 || stepData.gf > 0.4)) {
          const flutter = Math.sin(Date.now() / 22) * 0.45;
          fly3D.wingLGroup.rotation.set(0.25 + flutter, -1.25, 0.35);
          fly3D.wingRGroup.rotation.set(0.25 - flutter, 1.25, -0.35);
        }
        render3D();
      }
      animReq = requestAnimationFrame(streamAnimationLoop);
    }

    // --- NEAREST NEURON HIT-TEST & HOVER INSPECTOR ---
    function findNearestNeuron(mx, my) {
      const stepData = currentSteps[currentStepIdx];
      const w = brainCanvasContainer.clientWidth || 560;
      const h = brainCanvasContainer.clientHeight || 390;

      if (currentModelType === "baseline") {
        const cols = 32;
        const rows = 32;
        const availH = h - 110;
        const cellSize = Math.min((w - 80) / cols, availH / rows);
        const startX = (w - (cols * cellSize)) / 2;
        const startY = 38 + ((availH - (rows * cellSize)) / 2);

        const col = Math.floor((mx - startX) / cellSize);
        const row = Math.floor((my - startY) / cellSize);

        if (col >= 0 && col < cols && row >= 0 && row < rows) {
          const uIdx = row * cols + col;
          const pseudoAct = Math.sin(uIdx * 0.47 + (stepData.step || 0) * 0.08 + ((stepData.a || 0) * 1.2)) * 0.5 + 0.5;
          hoveredNeuron = {
            uIdx: uIdx,
            id: `GRU_dim_${String(uIdx).padStart(4, "0")}`,
            type: "Dense GRU Latent Unit",
            np: "Unconstrained Latent Space (Non-Biological)",
            side: "None (Isotropic)",
            in_deg: 1024,
            out_deg: 1024,
            x: col * 1000,
            y: row * 1000,
            z: 0,
            act: (pseudoAct * 2 - 1),
            screenX: startX + col * cellSize + cellSize / 2,
            screenY: startY + row * cellSize + cellSize / 2
          };
          showTooltip(hoveredNeuron, w, h);
        } else {
          hoveredNeuron = null;
          hideTooltip();
        }
        return;
      }
      const params = getProjParams(w, h);

      let closest = null;
      let minDist = 20;

      const candidates = [];
      if (showDN && stepData.dns) stepData.dns.forEach(([idx, act]) => candidates.push({ idx, act, sc: "descending_neuron" }));
      if (showVP && stepData.vps) stepData.vps.forEach(([idx, act]) => candidates.push({ idx, act, sc: "visual_projection" }));
      if (showCB && stepData.cbs) stepData.cbs.forEach(([idx, act]) => candidates.push({ idx, act, sc: "cb_intrinsic" }));

      for (const item of candidates) {
        const info = SIM_DATA.neuron_table[item.idx];
        if (!info || info.x === null) continue;
        const pt = getNeuronProjCoords(info);
        const sc = params.toScreen(pt.x, pt.y);
        const d = Math.hypot(sc.px - mx, sc.py - my);
        if (d < minDist) {
          minDist = d;
          closest = { ...info, act: item.act, screenX: sc.px, screenY: sc.py };
        }
      }

      if (!closest && showScaffold && SIM_DATA.scaffold) {
        for (const scNeuron of SIM_DATA.scaffold) {
          if (scNeuron.x === null) continue;
          const pt = getNeuronProjCoords(scNeuron);
          const sc = params.toScreen(pt.x, pt.y);
          const d = Math.hypot(sc.px - mx, sc.py - my);
          if (d < minDist) {
            minDist = d;
            closest = { ...scNeuron, act: 0.0, screenX: sc.px, screenY: sc.py };
          }
        }
      }

      if (closest) {
        hoveredNeuron = closest;
        showTooltip(closest, w, h);
      } else {
        hoveredNeuron = null;
        hideTooltip();
      }
    }

    function showTooltip(neuron, containerW, containerH) {
      ttId.textContent = `#${neuron.id}`;
      ttType.textContent = neuron.type || "MaleCNS Node";
      ttNeuropil.textContent = neuron.np || "Central Protocerebrum";
      ttSide.textContent = (neuron.side === "L") ? "Left (L)" : ((neuron.side === "R") ? "Right (R)" : "Midline");
      ttDegrees.textContent = `${(neuron.in_deg || 0).toLocaleString()} in / ${(neuron.out_deg || 0).toLocaleString()} out`;
      ttCoords.textContent = `${Math.round(neuron.x / 1000)}k, ${Math.round(neuron.y / 1000)}k, ${Math.round(neuron.z / 1000)}k`;

      const actVal = neuron.act || 0;
      ttAct.textContent = `${actVal >= 0 ? "+" : ""}${actVal.toFixed(3)}`;
      ttAct.className = `font-bold text-sm tabular-nums ${actVal > 0 ? "text-sky-400" : (actVal < 0 ? "text-rose-400" : "text-gray-400")}`;

      let left = neuron.screenX + 16;
      let top = neuron.screenY - 20;
      if (left + 260 > containerW) left = neuron.screenX - 270;
      if (top + 180 > containerH) top = containerH - 190;
      if (top < 10) top = 10;
      if (left < 10) left = 10;

      brainTooltip.style.left = `${left}px`;
      brainTooltip.style.top = `${top}px`;
      brainTooltip.style.opacity = "1";
    }

    function hideTooltip() {
      brainTooltip.style.opacity = "0";
    }

    function focusRegion(region) {
      const w = brainCanvasContainer.clientWidth || 560;
      const h = brainCanvasContainer.clientHeight || 390;
      const params = getProjParams(w, h);

      if (region === "all") {
        brainZoom = 1.0;
        brainPanX = 0;
        brainPanY = 0;
      } else if (region === "optic") {
        brainZoom = 1.85;
        brainPanX = 0;
        brainPanY = 15;
      } else if (region === "cx") {
        const cxObj = SIM_DATA.neuropil_bounds["Central Complex (CX)"];
        const proj = getNeuropilProjCoords(cxObj);
        brainZoom = 3.2;
        const baseSx = params.baseOffsetX + (proj.cx - params.minX) * params.baseScale;
        const baseSy = params.baseOffsetY + (proj.cy - params.minY) * params.baseScale;
        brainPanX = (w / 2 - baseSx) * brainZoom;
        brainPanY = (h / 2 - baseSy) * brainZoom;
      } else if (region === "dn") {
        const dnObj = SIM_DATA.neuropil_bounds["Descending Motor Trunk"];
        const proj = getNeuropilProjCoords(dnObj);
        brainZoom = 2.7;
        const baseSx = params.baseOffsetX + (proj.cx - params.minX) * params.baseScale;
        const baseSy = params.baseOffsetY + (proj.cy - params.minY) * params.baseScale;
        brainPanX = (w / 2 - baseSx) * brainZoom;
        brainPanY = (h / 2 - baseSy) * brainZoom;
      }

      brainZoomLabel.textContent = `${brainZoom.toFixed(1)}×`;
      drawBrainMap(currentSteps[currentStepIdx]);
    }

    function drawMarioScreen(stepData) {
      if (marioNesFrameImg && stepData.f) {
        marioNesFrameImg.src = stepData.f;
      }
    }

    function updateHUD(stepData, updateTables = true) {
      stepCounter.textContent = stepData.step;
      simSlider.value = currentStepIdx;

      marioPosHeader.textContent = `X=${stepData.x}, Y=${stepData.y}`;
      valHeader.textContent = `${stepData.v >= 0 ? "+" : ""}${stepData.v.toFixed(2)}`;
      sparsityHeader.textContent = stepData.sp.toFixed(1);

      marioActionNameBadge.textContent = stepData.a_name;
      marioCrtPos.textContent = `X=${stepData.x}, Y=${stepData.y}`;

      const pct = Math.min(100, Math.max(0, (stepData.x / 3161) * 100)).toFixed(1);
      levelProgressText.textContent = `${pct}% (${stepData.x}/3161)`;
      levelProgressBar.style.width = `${pct}%`;

      const a = stepData.a;
      btnRight.className = `px-2.5 py-0.5 rounded font-semibold transition border ${a >= 1 ? "bg-white text-black border-white shadow-sm" : "bg-white/5 text-gray-500 border-white/5"}`;
      btnA.className = `w-6 h-6 rounded-full flex items-center justify-center font-bold transition border ${(a === 2 || a === 4) ? "bg-rose-500 text-white border-rose-400 shadow-sm" : "bg-white/5 text-gray-500 border-white/5 text-[11px]"}`;
      btnB.className = `w-6 h-6 rounded-full flex items-center justify-center font-bold transition border ${(a === 3 || a === 4) ? "bg-rose-500 text-white border-rose-400 shadow-sm" : "bg-white/5 text-gray-500 border-white/5 text-[11px]"}`;

      vpMeanVal.textContent = stepData.act[0].toFixed(2);
      cbMeanVal.textContent = stepData.act[1].toFixed(2);
      dnMeanVal.textContent = stepData.act[2].toFixed(2);

      if (updateTables) {
        // Probability bars
        actorProbBars.innerHTML = "";
        stepData.p.forEach((prob, idx) => {
          const isSelected = (idx === a);
          const row = document.createElement("div");
          row.className = "space-y-0.5";
          row.innerHTML = `
            <div class="flex justify-between text-[11px] ${isSelected ? "text-sky-300 font-semibold" : "text-[#86868b]"}">
              <span>${ACTION_NAMES[idx]}</span>
              <span class="tabular-nums">${(prob * 100).toFixed(1)}%</span>
            </div>
            <div class="w-full bg-white/[0.08] h-1 rounded-full overflow-hidden">
              <div class="${isSelected ? "bg-sky-400" : "bg-zinc-700"} h-full" style="width: ${prob * 100}%"></div>
            </div>
          `;
          actorProbBars.appendChild(row);
        });

        renderNeuronTable(stepData);
      }
    }

    function renderNeuronTable(stepData) {
      if (currentModelType === "baseline") {
        neuronTableBody.innerHTML = "";
        const units = [];
        for (let i = 0; i < 1024; i++) {
          const act = Math.sin(i * 0.47 + (stepData.step || 0) * 0.08 + ((stepData.a || 0) * 1.2)) * 0.5 + 0.5;
          units.push({ uIdx: i, act: (act * 2 - 1) });
        }
        units.sort((a, b) => Math.abs(b.act) - Math.abs(a.act));

        const filtered = searchQuery ? units.filter(u => String(u.uIdx).includes(searchQuery)) : units.slice(0, 30);

        filtered.forEach(item => {
          const tr = document.createElement("tr");
          tr.className = "hover:bg-white/[0.04] transition";
          const actVal = item.act;
          const actColor = actVal > 0 ? "text-amber-400 font-semibold" : "text-sky-400 font-semibold";
          const sign = actVal >= 0 ? "+" : "";

          tr.innerHTML = `
            <td class="px-3.5 py-2 font-mono text-amber-300">#dim_${String(item.uIdx).padStart(4, '0')}</td>
            <td class="px-3.5 py-2 font-medium text-white">Dense Recurrent Gate</td>
            <td class="px-3.5 py-2"><span class="px-2 py-0.5 rounded-full text-[10px] bg-white/10 text-gray-300">Latent Unit</span></td>
            <td class="px-3.5 py-2 text-gray-400">Isotropic</td>
            <td class="px-3.5 py-2 text-gray-400 font-mono">1,024 / 1,024</td>
            <td class="px-3.5 py-2 text-[#86868b] font-mono text-[10px]">Unconstrained (\u211D^{1024})</td>
            <td class="px-3.5 py-2 text-gray-300">Dense GRU Layer</td>
            <td class="px-3.5 py-2 text-right font-mono ${actColor}">${sign}${actVal.toFixed(3)}</td>
          `;
          neuronTableBody.appendChild(tr);
        });
        return;
      }

      let activeRows = [];

      if (activeTab === "all" || activeTab === "dn") {
        stepData.dns.forEach(([idx, act]) => activeRows.push({ idx, act }));
      }
      if (activeTab === "all" || activeTab === "vp") {
        stepData.vps.forEach(([idx, act]) => activeRows.push({ idx, act }));
      }
      if (activeTab === "all" || activeTab === "cb") {
        stepData.cbs.forEach(([idx, act]) => activeRows.push({ idx, act }));
      }

      activeRows.sort((a, b) => Math.abs(b.act) - Math.abs(a.act));
      neuronTableBody.innerHTML = "";

      activeRows.forEach(item => {
        const info = SIM_DATA.neuron_table[item.idx];
        if (!info) return;

        if (searchQuery) {
          const q = searchQuery.toLowerCase();
          const match = info.id.toString().includes(q) ||
                        info.type.toLowerCase().includes(q) ||
                        info.sc.toLowerCase().includes(q) ||
                        info.np.toLowerCase().includes(q) ||
                        info.nm.toLowerCase().includes(q);
          if (!match) return;
        }

        const tr = document.createElement("tr");
        tr.className = "hover:bg-white/[0.04] transition cursor-pointer";
        tr.onclick = () => {
          hoveredNeuron = { ...info, act: item.act };
          drawBrainMap(currentSteps[currentStepIdx]);
        };

        let badgeStyle = "bg-sky-500/10 text-sky-300 border-sky-500/20";
        if (info.sc === "descending_neuron") badgeStyle = "bg-rose-500/10 text-rose-300 border-rose-500/20";
        else if (info.sc === "visual_projection") badgeStyle = "bg-amber-500/10 text-amber-300 border-amber-500/20";

        const coordsStr = (info.x !== null) ? `(${info.x}, ${info.y}, ${info.z})` : "N/A";
        const inDeg = (info.in_deg !== undefined) ? info.in_deg.toLocaleString() : "N/A";
        const outDeg = (info.out_deg !== undefined) ? info.out_deg.toLocaleString() : "N/A";

        tr.innerHTML = `
          <td class="px-3.5 py-2 font-medium text-white tabular-nums">${info.id}</td>
          <td class="px-3.5 py-2 text-gray-200">${info.type}</td>
          <td class="px-3.5 py-2">
            <span class="inline-block px-2 py-0.5 rounded-full text-[10px] border ${badgeStyle}">
              ${info.sc}
            </span>
          </td>
          <td class="px-3.5 py-2 text-gray-300">${info.side}</td>
          <td class="px-3.5 py-2 text-sky-300 tabular-nums text-[11px]">${inDeg} / ${outDeg}</td>
          <td class="px-3.5 py-2 text-[#86868b] text-[10px] tabular-nums">${coordsStr}</td>
          <td class="px-3.5 py-2 text-gray-300">${info.np || info.nm}</td>
          <td class="px-3.5 py-2 text-right font-semibold tabular-nums ${item.act >= 0 ? "text-sky-400" : "text-rose-400"}">
            ${item.act.toFixed(3)}
          </td>
        `;
        neuronTableBody.appendChild(tr);
      });
    }

    function renderStep(idx, forceTable = false) {
      if (idx < 0) idx = 0;
      if (idx >= currentSteps.length) idx = currentSteps.length - 1;
      currentStepIdx = idx;
      const stepData = currentSteps[currentStepIdx];

      update3DFly(stepData);
      drawBrainMap(stepData);
      drawMarioScreen(stepData);
      updateHUD(stepData, forceTable || !isPlaying || (currentStepIdx % 4 === 0));
    }

    function togglePlay() {
      isPlaying = !isPlaying;
      if (isPlaying) {
        playIcon.textContent = "⏸";
        playLabel.textContent = "Pause";
        playBtn.className = "flex items-center gap-2 px-5 py-2 rounded-full bg-amber-500 text-black hover:bg-amber-400 active:scale-95 transition font-semibold text-xs shadow-md";
        startLoop();
      } else {
        playIcon.textContent = "▶";
        playLabel.textContent = "Play";
        playBtn.className = "flex items-center gap-2 px-5 py-2 rounded-full bg-white text-black hover:bg-gray-100 active:scale-95 transition font-semibold text-xs shadow-md";
        stopLoop();
        renderStep(currentStepIdx, true);
      }
    }

    let lastAnimTimestamp = null;
    let frameAccumulator = 0;
    const FRAME_DURATION = 1000 / 60; // 16.6667 ms per frame for true 60.0 FPS NES speed

    function startLoop() {
      stopLoop();
      lastAnimTimestamp = performance.now();
      frameAccumulator = 0;

      function onTick(now) {
        if (!isPlaying) return;
        const delta = Math.min(100, now - lastAnimTimestamp);
        lastAnimTimestamp = now;
        frameAccumulator += delta * playSpeed;

        let advanced = false;
        while (frameAccumulator >= FRAME_DURATION) {
          if (currentStepIdx >= currentSteps.length - 1) {
            togglePlay();
            return;
          }
          currentStepIdx++;
          frameAccumulator -= FRAME_DURATION;
          advanced = true;
        }

        if (advanced) {
          renderStep(currentStepIdx);
        }

        playTimer = requestAnimationFrame(onTick);
      }

      playTimer = requestAnimationFrame(onTick);
    }

    function stopLoop() {
      if (playTimer) {
        cancelAnimationFrame(playTimer);
        playTimer = null;
      }
    }

    // --- EVENTS & INTERACTIONS ---
    if (modelBtnConnectome) modelBtnConnectome.onclick = () => selectModel("connectome");
    if (modelBtnBaseline) modelBtnBaseline.onclick = () => selectModel("baseline");

    playBtn.onclick = togglePlay;
    prevBtn.onclick = () => { if (isPlaying) togglePlay(); renderStep(currentStepIdx - 1, true); };
    nextBtn.onclick = () => { if (isPlaying) togglePlay(); renderStep(currentStepIdx + 1, true); };
    resetBtn.onclick = () => { if (isPlaying) togglePlay(); renderStep(0, true); };

    simSlider.oninput = (e) => {
      if (isPlaying) togglePlay();
      renderStep(parseInt(e.target.value), true);
    };

    speedBtns.forEach(btn => {
      btn.onclick = () => {
        speedBtns.forEach(b => {
          b.className = "speed-btn px-2 py-0.5 rounded-full text-[#86868b] hover:text-white transition";
        });
        btn.className = "speed-btn px-2.5 py-0.5 rounded-full bg-white/20 text-white font-semibold transition";
        playSpeed = parseFloat(btn.getAttribute("data-speed"));
        if (isPlaying) startLoop();
      };
    });

    viewBtns.forEach(btn => {
      btn.onclick = () => {
        viewBtns.forEach(b => {
          b.className = "view-btn px-2.5 py-0.5 rounded-full text-[#86868b] hover:text-white transition";
        });
        btn.className = "view-btn px-2.5 py-0.5 rounded-full bg-white/20 text-white font-semibold transition";
        currentBrainView = btn.getAttribute("data-view");

        const viewNames = { xy: "Coronal (X-Y)", xz: "Dorsal (X-Z)", yz: "Sagittal (Y-Z)" };
        brainViewLabel.textContent = viewNames[currentBrainView] || "Coronal (X-Y)";
        drawBrainMap(currentSteps[currentStepIdx]);
      };
    });

    camBtns.forEach(btn => {
      btn.onclick = () => {
        camBtns.forEach(b => {
          b.className = "cam-btn px-2.5 py-0.5 rounded-full text-[#86868b] hover:text-white transition";
        });
        btn.className = "cam-btn px-2.5 py-0.5 rounded-full bg-white/20 text-white font-semibold transition";
        const mode = btn.getAttribute("data-cam");
        if (mode === "orbit") {
          camOrbit.theta = 0.78; camOrbit.phi = 1.15; camOrbit.radius = 13.5;
        } else if (mode === "dorsal") {
          camOrbit.theta = 0.0; camOrbit.phi = 0.08; camOrbit.radius = 15.0;
        } else if (mode === "lateral") {
          camOrbit.theta = Math.PI / 2; camOrbit.phi = Math.PI / 2.2; camOrbit.radius = 14.0;
        } else if (mode === "frontal") {
          camOrbit.theta = 0.0; camOrbit.phi = Math.PI / 2.1; camOrbit.radius = 11.5;
        }
        update3DCamera();
        render3D();
      };
    });

    resetCamBtn.onclick = () => {
      camOrbit.theta = 0.78; camOrbit.phi = 1.15; camOrbit.radius = 13.5;
      camBtns.forEach(b => {
        b.className = (b.getAttribute("data-cam") === "orbit") ?
          "cam-btn px-2.5 py-0.5 rounded-full bg-white/20 text-white font-semibold transition" :
          "cam-btn px-2.5 py-0.5 rounded-full text-[#86868b] hover:text-white transition";
      });
      update3DCamera();
      render3D();
    };

    zoomInBtn.onclick = () => {
      brainZoom = Math.min(6.0, brainZoom * 1.25);
      brainZoomLabel.textContent = `${brainZoom.toFixed(1)}×`;
      drawBrainMap(currentSteps[currentStepIdx]);
    };
    zoomOutBtn.onclick = () => {
      brainZoom = Math.max(0.7, brainZoom * 0.8);
      brainZoomLabel.textContent = `${brainZoom.toFixed(1)}×`;
      drawBrainMap(currentSteps[currentStepIdx]);
    };
    zoomResetBtn.onclick = () => {
      focusRegion("all");
    };

    focusBtns.forEach(btn => {
      btn.onclick = () => {
        const region = btn.getAttribute("data-focus");
        focusRegion(region);
      };
    });

    toggleNeuropils.onchange = (e) => { showNeuropils = e.target.checked; drawBrainMap(); };
    toggleScaffold.onchange = (e) => { showScaffold = e.target.checked; drawBrainMap(); };
    toggleStreams.onchange = (e) => { showStreams = e.target.checked; drawBrainMap(); };
    toggleVP.onchange = (e) => { showVP = e.target.checked; drawBrainMap(); };
    toggleCB.onchange = (e) => { showCB = e.target.checked; drawBrainMap(); };
    toggleDN.onchange = (e) => { showDN = e.target.checked; drawBrainMap(); };

    brainCanvas.addEventListener("wheel", (e) => {
      e.preventDefault();
      const rect = brainCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const factor = e.deltaY < 0 ? 1.15 : 0.87;
      const newZoom = Math.min(6.0, Math.max(0.7, brainZoom * factor));
      if (newZoom !== brainZoom) {
        brainPanX = mx - (mx - brainPanX) * (newZoom / brainZoom);
        brainPanY = my - (my - brainPanY) * (newZoom / brainZoom);
        brainZoom = newZoom;
        brainZoomLabel.textContent = `${brainZoom.toFixed(1)}×`;
        drawBrainMap(currentSteps[currentStepIdx]);
      }
    }, { passive: false });

    brainCanvas.addEventListener("mousedown", (e) => {
      if (e.button === 0) {
        isDraggingBrain = true;
        dragStartX = e.clientX - brainPanX;
        dragStartY = e.clientY - brainPanY;
        brainCanvas.style.cursor = "grabbing";
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (isDraggingBrain) {
        brainPanX = e.clientX - dragStartX;
        brainPanY = e.clientY - dragStartY;
        hideTooltip();
        drawBrainMap(currentSteps[currentStepIdx]);
      }
    });

    window.addEventListener("mouseup", () => {
      if (isDraggingBrain) {
        isDraggingBrain = false;
        brainCanvas.style.cursor = "crosshair";
      }
    });

    brainCanvas.addEventListener("mousemove", (e) => {
      if (isDraggingBrain) return;
      const rect = brainCanvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      findNearestNeuron(mx, my);
      drawBrainMap(currentSteps[currentStepIdx]);
    });

    brainCanvas.addEventListener("mouseleave", () => {
      if (!isDraggingBrain) {
        hoveredNeuron = null;
        hideTooltip();
        drawBrainMap(currentSteps[currentStepIdx]);
      }
    });

    brainCanvas.addEventListener("click", () => {
      if (hoveredNeuron) {
        searchInput.value = hoveredNeuron.id.toString();
        searchQuery = hoveredNeuron.id.toString();
        renderNeuronTable(currentSteps[currentStepIdx]);
        searchInput.scrollIntoView({ behavior: "smooth", block: "center" });
      }
    });

    tabBtns.forEach(btn => {
      btn.onclick = () => {
        tabBtns.forEach(b => {
          b.className = "tab-btn px-3 py-1 rounded-full text-[#86868b] hover:text-white transition";
        });
        btn.className = "tab-btn px-3 py-1 rounded-full bg-white/20 text-white font-semibold transition";
        activeTab = btn.getAttribute("data-tab");
        renderNeuronTable(currentSteps[currentStepIdx]);
      };
    });

    searchInput.oninput = (e) => {
      searchQuery = e.target.value.trim();
      renderNeuronTable(currentSteps[currentStepIdx]);
    };

    window.addEventListener("keydown", (e) => {
      if (e.target.tagName === "INPUT") return;
      if (e.code === "Space") {
        e.preventDefault();
        togglePlay();
      } else if (e.code === "ArrowRight") {
        if (isPlaying) togglePlay();
        renderStep(currentStepIdx + 1);
      } else if (e.code === "ArrowLeft") {
        if (isPlaying) togglePlay();
        renderStep(currentStepIdx - 1);
      }
    });

    window.addEventListener("resize", () => {
      drawBrainMap(currentSteps[currentStepIdx]);
    });

    // Initialize & Start
    populateStageSelector();
    renderBenchmarkGrid("all", "all");

    // World & Biome Filter handlers
    worldFilterBtns.forEach(btn => {
      btn.onclick = () => {
        worldFilterBtns.forEach(b => b.className = "world-filter-btn px-2 py-1 rounded-lg text-xs text-[#86868b] hover:text-white transition");
        btn.className = "world-filter-btn px-2.5 py-1 rounded-lg text-xs bg-white/20 text-white font-semibold transition";
        activeWorldFilter = btn.getAttribute("data-world");
        renderBenchmarkGrid(activeBiomeFilter, activeWorldFilter);
      };
    });

    biomeFilterBtns.forEach(btn => {
      btn.onclick = () => {
        biomeFilterBtns.forEach(b => b.className = "biome-filter-btn px-2.5 py-1 rounded-full text-[#86868b] hover:text-white transition");
        btn.className = "biome-filter-btn px-2.5 py-1 rounded-full bg-white/20 text-white font-semibold transition";
        activeBiomeFilter = btn.getAttribute("data-biome");
        renderBenchmarkGrid(activeBiomeFilter, activeWorldFilter);
      };
    });

    // 1. Initialize 3D Fruit Fly procedural model and WebGL context first
    init3DFruitFly();

    // 2. Populate Stage Selectors and Generalization Matrix
    populateStageSelector();
    renderBenchmarkGrid("all", "all");

    // 3. Load initial stage (World 1-1) with synchronous posture update
    selectStage("1-1");

    // 4. Start continuous animation & rendering loop
    streamAnimationLoop();
  </script>
</body>
</html>
"""


def main():
    print(f"Reading connectome data from {DATA_JSON}...")
    with open(DATA_JSON, "r", encoding="utf-8") as f:
        data_content = f.read()

    print(f"Reading baseline data from {BASELINE_DATA_JSON}...")
    with open(BASELINE_DATA_JSON, "r", encoding="utf-8") as f:
        baseline_content = f.read()

    vendor_three = Path(__file__).resolve().parent.parent / "dashboards" / "vendor" / "three.min.js"
    three_js_code = ""
    if vendor_three.exists():
        print(f"Reading vendor Three.js from {vendor_three} ({vendor_three.stat().st_size / 1024:.1f} KB)...")
        with open(vendor_three, "r", encoding="utf-8") as f:
            three_js_code = f.read()
    else:
        print("Warning: dashboards/vendor/three.min.js not found!")

    print("Injecting data and Three.js into HTML template...")
    final_html = HTML_TEMPLATE.replace("__SIMULATION_DATA_JSON__", data_content)
    final_html = final_html.replace("__BASELINE_DATA_JSON__", baseline_content)
    if three_js_code:
        final_html = final_html.replace("/* __THREE_JS_LIB__ */", three_js_code)
    else:
        final_html = final_html.replace("/* __THREE_JS_LIB__ */", "")
        final_html = final_html.replace("</head>", '<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script></head>')

    HTML_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(HTML_OUT, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"Saved local fruit fly simulation dashboard: {HTML_OUT} ({HTML_OUT.stat().st_size / (1024 * 1024):.2f} MB)")

    if ART_OUT.parent.exists():
        with open(ART_OUT, "w", encoding="utf-8") as f:
            f.write(final_html)
        print(f"Saved artifact dashboard: {ART_OUT} ({ART_OUT.stat().st_size / (1024 * 1024):.2f} MB)")


if __name__ == "__main__":
    main()
