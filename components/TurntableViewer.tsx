"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { resolveModel, type ModelSpec } from "@/lib/models";
import {
  DEFAULT_SETTINGS,
  DESKTOP_LAYOUT,
  MOBILE_LAYOUT,
  startViewer,
  type ViewerHandle,
  type ViewerInfo,
  type ViewerProgress,
  type ViewerSettings,
  type ViewerState,
} from "@/lib/render/engine";
import styles from "./viewer.module.css";

const PHASE_LABEL: Record<ViewerProgress["phase"], string> = {
  gpu: "Initializing WebGPU",
  download: "Downloading model",
  parse: "Reading glTF geometry",
  environment: "Baking studio lighting",
  textures: "Uploading textures and mipmaps",
  geometry: "Building vertex buffers",
  compile: "Compiling WGSL pipelines",
  ready: "Ready",
};

/**
 * The stacked "table" layout (deck on top, album below) is the default everywhere.
 * `?layout=studio` brings back the dark orbit-camera set with the full control panel.
 */
function useMobileLayout(): boolean | undefined {
  const [mobile, setMobile] = useState<boolean | undefined>(undefined);
  useEffect(() => {
    const studio = new URLSearchParams(window.location.search).get("layout") === "studio";
    const timer = setTimeout(() => setMobile(!studio), 0);
    return () => clearTimeout(timer);
  }, []);
  return mobile;
}

/** Phones and tablets: coarse pointer without hover. `?quality=mobile|high` overrides. */
function isTouchDevice(): boolean {
  if (typeof window === "undefined") return false;
  const forced = new URLSearchParams(window.location.search).get("quality");
  if (forced === "mobile") return true;
  if (forced === "high") return false;
  return window.matchMedia("(pointer: coarse) and (hover: none)").matches;
}

function formatBytes(n: number): string {
  if (n > 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n > 1e3) return `${(n / 1e3).toFixed(0)} KB`;
  return `${n} B`;
}

export default function TurntableViewer({ modelId }: { modelId?: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const handleRef = useRef<ViewerHandle | null>(null);
  const [model, setModel] = useState<ModelSpec>(() => resolveModel(modelId));
  const [modelResolved, setModelResolved] = useState(false);
  const [settings, setSettings] = useState<ViewerSettings>(DEFAULT_SETTINGS);
  const [progress, setProgress] = useState<ViewerProgress>({ phase: "gpu" });
  const [info, setInfo] = useState<ViewerInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  const [deck, setDeck] = useState<ViewerState>({ playing: true, armDown: true, armMoving: false, pitch: 0, audioEnabled: false, progress: 0, speed: 33, quartz: false, qualityLevel: 0, qualityLevels: 6, qualityMode: "auto" });
  const [menuOpen, setMenuOpen] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const spectrumRef = useRef<HTMLCanvasElement>(null);
  const [hudHidden, setHudHidden] = useState(false);
  const mobile = useMobileLayout();

  // Allow ?model=helmet for pipeline checks without touching the default.
  // Resolved once on the client (search params are not available during SSR).
  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("model");
    const timer = setTimeout(() => {
      if (id) setModel(resolveModel(id));
      setModelResolved(true);
    }, 0);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !modelResolved || mobile === undefined) return;
    setInfo(null);
    setError(null);
    setProgress({ phase: "gpu" });
    const handle = startViewer(canvas, model, settings, {
      onProgress: setProgress,
      onReady: setInfo,
      onError: (e) => setError(e.message),
      onStats: ({ fps }) => setFps(fps),
      onState: setDeck,
    }, mobile ? { ...MOBILE_LAYOUT, quality: isTouchDevice() ? "mobile" : "high" } : DESKTOP_LAYOUT);
    handleRef.current = handle;
    return () => {
      handleRef.current = null;
      handle.dispose();
    };
    // settings are pushed through handle.update below; only the model restarts the engine.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model, modelResolved, mobile]);

  const update = useCallback((partial: Partial<ViewerSettings>) => {
    setSettings((prev) => {
      const next = { ...prev, ...partial };
      handleRef.current?.update(partial);
      return next;
    });
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "h" || e.key === "H") setHudHidden((v) => !v);
      if (e.key === "r" || e.key === "R") handleRef.current?.resetCamera();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Spectrum visualizer: pulls analyser data from the engine every frame.
  useEffect(() => {
    if (!deck.audioEnabled) return;
    const canvas = spectrumRef.current;
    if (!canvas) return;
    const ctx2d = canvas.getContext("2d");
    if (!ctx2d) return;
    const bins = new Uint8Array(256);
    let raf = 0;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      if (!handleRef.current?.spectrum(bins)) return;
      const { width, height } = canvas;
      ctx2d.clearRect(0, 0, width, height);
      const bars = 48;
      const gap = 2;
      const barWidth = (width - gap * (bars - 1)) / bars;
      for (let i = 0; i < bars; i++) {
        // Log-spaced bins (≈40 Hz – 12 kHz) so the bass does not hog the display.
        const lo = 1, hi = 128;
        const start = Math.floor(lo * Math.pow(hi / lo, i / bars));
        const end = Math.max(start + 1, Math.floor(lo * Math.pow(hi / lo, (i + 1) / bars)));
        let peak = 0;
        for (let b = start; b < end; b++) peak = Math.max(peak, bins[b]);
        const h = Math.max(2, (peak / 255) * height);
        ctx2d.fillStyle = `rgba(230, 215, 178, ${0.35 + 0.65 * (peak / 255)})`;
        ctx2d.fillRect(i * (barWidth + gap), height - h, barWidth, h);
      }
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [deck.audioEnabled]);

  const loading = !info && !error;
  const percent = useMemo(() => {
    if (progress.phase === "download" && progress.total) return (progress.loaded! / progress.total) * 100;
    if (progress.phase === "textures" && progress.total) return (progress.loaded! / progress.total) * 100;
    return undefined;
  }, [progress]);

  // Mobile has no buttons: the first tap anywhere starts the audio (browser autoplay policy).
  useEffect(() => {
    if (!mobile || !model.audio || deck.audioEnabled || !info) return;
    const start = () => { void handleRef.current?.enableAudio(); };
    window.addEventListener("pointerup", start, { once: true });
    window.addEventListener("keydown", start, { once: true });
    return () => {
      window.removeEventListener("pointerup", start);
      window.removeEventListener("keydown", start);
    };
  }, [mobile, model.audio, deck.audioEnabled, info]);

  if (mobile) {
    const cover = model.label?.image;
    return (
      <div className={styles.mobileRoot}>
        <div className={styles.mobileStage}>
          <canvas ref={canvasRef} className={styles.canvas} aria-label={`3D view of ${model.title}`} />
          {loading && (
            <div className={styles.mobileOverlay} role="status" aria-live="polite">
              <div className={styles.mobileLoaderPhase}>{PHASE_LABEL[progress.phase]}{progress.phase === "download" && progress.loaded !== undefined ? ` · ${formatBytes(progress.loaded)}` : ""}</div>
              <div className={styles.mobileBar}><div className={percent === undefined ? styles.barIndeterminate : styles.barFill} style={percent !== undefined ? { width: `${percent}%` } : undefined} /></div>
            </div>
          )}
          {error && (
            <div className={styles.mobileOverlay} role="alert">
              <p className={styles.errorText}>{error}</p>
            </div>
          )}
        </div>

        <section className={styles.album} aria-label="Album">
          {cover && <img className={styles.cover} src={cover} alt="" width={96} height={96} />}
          <h1 className={styles.albumTitle}>{model.audio?.title ?? model.title}</h1>
        </section>

        <div className={styles.menuWrap}>
          <button type="button" className={styles.menuPill} onClick={() => setMenuOpen((v) => !v)} aria-expanded={menuOpen} aria-label="Quality and audio">
            <span className={styles.qualityBar} aria-hidden="true">
              {Array.from({ length: deck.qualityLevels }, (_, i) => (
                <span key={i} className={i <= deck.qualityLevel ? styles.qualitySegOn : styles.qualitySeg} />
              ))}
            </span>
            <span className={styles.menuPillText}>
              {deck.qualityMode === "auto" ? "Auto" : deck.qualityMode === "low" ? "Low" : deck.qualityMode === "medium" ? "Medium" : "High"}
            </span>
          </button>
          {menuOpen && (
            <div className={styles.menu} role="menu">
              <div className={styles.menuLabel}>Quality</div>
              <div className={styles.menuRow}>
                {(["auto", "low", "medium", "high"] as const).map((mode) => (
                  <button key={mode} type="button" role="menuitemradio" aria-checked={deck.qualityMode === mode}
                    className={deck.qualityMode === mode ? styles.menuBtnActive : styles.menuBtn}
                    onClick={() => handleRef.current?.setQualityMode(mode)}>
                    {mode === "auto" ? "Auto" : mode === "low" ? "Low" : mode === "medium" ? "Medium" : "High"}
                  </button>
                ))}
              </div>
              <div className={styles.menuHint}>Level {deck.qualityLevel + 1}/{deck.qualityLevels} · Auto keeps ≥30 fps</div>
              <div className={styles.menuLabel}>Audio</div>
              <div className={styles.menuRow}>
                <button type="button" className={styles.menuBtn} onClick={() => fileRef.current?.click()}>
                  🎵 Load a song…
                </button>
                {!deck.audioEnabled && model.audio && (
                  <button type="button" className={styles.menuBtn} onClick={() => handleRef.current?.enableAudio()}>🔊 Sound</button>
                )}
              </div>
              <div className={styles.menuHint}>{deck.audioEnabled ? "Playing through the stylus" : "The demo track is not on the public site; load your own audio file."}</div>
              <input ref={fileRef} type="file" accept="audio/*" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void handleRef.current?.loadAudioFile(f); e.target.value = ""; }} />
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className={styles.root}>
      <canvas ref={canvasRef} className={styles.canvas} aria-label={`3D view of ${model.title}`} />

      {loading && (
        <div className={styles.overlay} role="status" aria-live="polite">
          <div className={styles.loader}>
            <div className={styles.loaderTitle}>{model.title}</div>
            <div className={styles.loaderPhase}>
              {PHASE_LABEL[progress.phase]}
              {progress.phase === "download" && progress.loaded !== undefined && (
                <span className={styles.loaderDetail}>
                  {progress.detail ? ` · ${progress.detail}` : ""} · {formatBytes(progress.loaded)}
                  {progress.total ? ` / ${formatBytes(progress.total)}` : ""}
                </span>
              )}
              {progress.phase === "textures" && progress.total !== undefined && (
                <span className={styles.loaderDetail}> · {progress.loaded} / {progress.total}</span>
              )}
            </div>
            <div className={styles.bar}>
              <div
                className={percent === undefined ? styles.barIndeterminate : styles.barFill}
                style={percent !== undefined ? { width: `${percent}%` } : undefined}
              />
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className={styles.overlay} role="alert">
          <div className={styles.loader}>
            <div className={styles.loaderTitle}>The viewer could not start</div>
            <p className={styles.errorText}>{error}</p>
            <p className={styles.errorHint}>
              The viewer runs on WebGPU through <code>vgpu</code>. Check <code>chrome://gpu</code> and that the
              model exists at <code>{model.url}</code>.
            </p>
          </div>
        </div>
      )}

      {!hudHidden && (
        <>
          <header className={styles.header}>
            <div className={styles.eyebrow}>WebGPU · vgpu · Next.js</div>
            <h1 className={styles.title}>{model.title}</h1>
            <p className={styles.subtitle}>{model.subtitle}</p>
          </header>

          <aside className={styles.panel} aria-label="Render controls">
            <div className={styles.panelSection}>
              <div className={styles.panelHeading}>Finish</div>
              <div className={styles.segmented} role="radiogroup" aria-label="Finish">
                <button
                  type="button"
                  role="radio"
                  aria-checked={settings.finish === "black"}
                  className={settings.finish === "black" ? styles.segmentActive : styles.segment}
                  onClick={() => update({ finish: "black" })}
                  disabled={!model.recolorable}
                >
                  Black
                </button>
                <button
                  type="button"
                  role="radio"
                  aria-checked={settings.finish === "original"}
                  className={settings.finish === "original" ? styles.segmentActive : styles.segment}
                  onClick={() => update({ finish: "original" })}
                >
                  Original
                </button>
              </div>
            </div>

            <div className={styles.panelSection}>
              <Slider label="Exposure" value={settings.exposure} min={0.2} max={3} step={0.05} onChange={(v) => update({ exposure: v })} format={(v) => `${v.toFixed(2)}×`} />
              <Slider label="Rotate lighting" value={settings.envRotation} min={-Math.PI} max={Math.PI} step={0.01} onChange={(v) => update({ envRotation: v })} format={(v) => `${Math.round((v * 180) / Math.PI)}°`} />
              <Slider label="Ambient light" value={settings.envIntensity} min={0} max={2.5} step={0.05} onChange={(v) => update({ envIntensity: v })} format={(v) => v.toFixed(2)} />
              <Slider label="Key light" value={settings.lightIntensity} min={0} max={8} step={0.1} onChange={(v) => update({ lightIntensity: v })} format={(v) => v.toFixed(1)} />
              <Slider label="Bloom" value={settings.bloom} min={0} max={1.5} step={0.05} onChange={(v) => update({ bloom: v })} format={(v) => v.toFixed(2)} />
            </div>

            <div className={styles.panelSection}>
              <div className={styles.panelHeading}>Deck</div>
              <div className={styles.segmented}>
                <button type="button" className={deck.playing ? styles.segmentActive : styles.segment} onClick={() => handleRef.current?.togglePlay()} disabled={info ? info.spinningNodes.length === 0 : false}>
                  {deck.playing ? "■ Stop" : "▶ Start"}
                </button>
                <button type="button" className={deck.armDown ? styles.segmentActive : styles.segment} onClick={() => handleRef.current?.toggleArm()} disabled={info ? !info.hasTonearm : false}>
                  {deck.armDown ? "Arm: on record" : "Arm: at rest"}
                </button>
              </div>
              <Slider
                label="Pitch"
                value={deck.pitch * 100}
                min={-8}
                max={8}
                step={0.1}
                onChange={(v) => handleRef.current?.setPitch(v / 100)}
                format={(v) => `${v > 0 ? "+" : ""}${v.toFixed(1)}% · ${(33.33 * (1 + v / 100)).toFixed(2)} rpm`}
              />
              <Toggle label="Auto-rotate camera" checked={settings.autoRotate} onChange={(v) => update({ autoRotate: v })} />
            </div>

            {model.audio && (
              <div className={styles.panelSection}>
                <div className={styles.panelHeading}>Audio</div>
                <div className={styles.track}>
                  <span className={styles.trackTitle}>{model.audio.title}</span>
                  <span className={styles.trackArtist}>{model.audio.artist}</span>
                </div>
                {deck.audioEnabled ? (
                  <>
                    <canvas ref={spectrumRef} className={styles.spectrum} width={240} height={44} aria-label="Audio spectrum" />
                    <div className={styles.progress} aria-hidden="true">
                      <div className={styles.progressFill} style={{ width: `${deck.progress * 100}%` }} />
                    </div>
                  </>
                ) : (
                  <button type="button" className={styles.button} onClick={() => handleRef.current?.enableAudio()}>
                    🔊 Enable sound
                  </button>
                )}
              </div>
            )}

            <div className={styles.panelActions}>
              <button type="button" className={styles.button} onClick={() => handleRef.current?.resetCamera()}>
                Reset camera <kbd>R</kbd>
              </button>
              <button type="button" className={styles.buttonGhost} onClick={() => setHudHidden(true)}>
                Hide UI <kbd>H</kbd>
              </button>
            </div>
          </aside>

          <footer className={styles.footer}>
            <div className={styles.hint}>Drag to orbit · Wheel to zoom · Tap the arm, START/STOP or the pitch</div>
            {info && (
              <dl className={styles.stats}>
                <div><dt>FPS</dt><dd>{fps ? fps.toFixed(0) : "–"}</dd></div>
                <div><dt>Triangles</dt><dd>{info.triangles.toLocaleString("es")}</dd></div>
                <div><dt>Draw calls</dt><dd>{info.drawCalls}</dd></div>
                <div><dt>Textures</dt><dd>{info.textures}</dd></div>
                <div className={styles.statWide}><dt>GPU</dt><dd title={info.adapter}>{info.adapter}</dd></div>
              </dl>
            )}
            <div className={styles.credit}>
              {[model.credit, ...(model.attachments ?? []).map((a) => a.credit)].map((credit, i) => (
                <div key={credit.url}>
                  {i === 0 ? "Modelo: " : "+ "}
                  <a href={credit.url} target="_blank" rel="noopener noreferrer">
                    {credit.source}
                  </a>{" "}
                  por {credit.author} · {credit.license}
                </div>
              ))}
            </div>
          </footer>
        </>
      )}
      {hudHidden && (
        <button type="button" className={styles.showHud} onClick={() => setHudHidden(false)}>
          Show UI <kbd>H</kbd>
        </button>
      )}
    </div>
  );
}

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange(v: number): void;
  format(v: number): string;
}) {
  return (
    <label className={styles.slider}>
      <span className={styles.sliderLabel}>
        {props.label}
        <span className={styles.sliderValue}>{props.format(props.value)}</span>
      </span>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </label>
  );
}

function Toggle(props: { label: string; checked: boolean; onChange(v: boolean): void; disabled?: boolean }) {
  return (
    <label className={styles.toggle} aria-disabled={props.disabled}>
      <input type="checkbox" checked={props.checked} disabled={props.disabled} onChange={(e) => props.onChange(e.target.checked)} />
      <span className={styles.toggleTrack} aria-hidden="true"><span className={styles.toggleThumb} /></span>
      <span>{props.label}</span>
    </label>
  );
}
