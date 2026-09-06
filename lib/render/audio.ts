// Web Audio "cartridge": the track plays only while the stylus is down, and its playback
// rate follows the platter speed, so spin-up, spin-down and the pitch fader are audible.

export interface DeckAudioOptions {
  url: string;
  onEnded?(): void;
}

export class DeckAudio {
  private context?: AudioContext;
  private buffer?: AudioBuffer;
  private encoded?: ArrayBuffer;
  private source?: AudioBufferSourceNode;
  private gain?: GainNode;
  private analyser?: AnalyserNode;
  private needleDown = false;
  private rate = 0;
  private endedFired = false;
  /** Seconds into the track (integrated from the platter speed). */
  position = 0;
  loading: Promise<void>;

  constructor(private readonly options: DeckAudioOptions) {
    this.loading = fetch(options.url).then(async (r) => {
      if (!r.ok) throw new Error(`Audio ${r.status} at ${options.url}`);
      this.encoded = await r.arrayBuffer();
    });
  }

  get enabled(): boolean {
    return !!this.buffer;
  }

  get duration(): number {
    return this.buffer?.duration ?? 0;
  }

  /** Must run inside a user gesture (autoplay policy). */
  async enable(): Promise<void> {
    if (this.buffer) {
      await this.context?.resume();
      return;
    }
    try {
      await this.loading;
    } catch (error) {
      console.warn("[viewer] audio track not available on this deployment:", error);
      return;
    }
    const context = new AudioContext();
    this.context = context;
    this.gain = context.createGain();
    this.gain.gain.value = 0;
    this.analyser = context.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.82;
    this.gain.connect(this.analyser).connect(context.destination);
    this.buffer = await context.decodeAudioData(this.encoded!.slice(0));
    await context.resume();
    this.restart();
  }

  /** (Re)creates the source at the current position; needed after a natural end or a seek. */
  private restart(): void {
    if (!this.context || !this.buffer || !this.gain) return;
    this.source?.stop();
    const source = this.context.createBufferSource();
    source.buffer = this.buffer;
    source.playbackRate.value = Math.max(this.rate, 0.0001);
    source.connect(this.gain);
    source.start(0, Math.min(this.position, this.buffer.duration - 0.01));
    this.source = source;
    this.endedFired = false;
  }

  /** Rewind to the lead-in groove. */
  seek(seconds: number): void {
    this.position = Math.max(0, Math.min(seconds, this.duration || Infinity));
    if (this.buffer) this.restart();
  }

  setNeedleDown(down: boolean): void {
    if (down === this.needleDown) return;
    this.needleDown = down;
    this.applyGain();
  }

  /** Platter speed relative to nominal (1 = 33⅓ rpm). */
  setRate(rate: number): void {
    this.rate = rate;
    if (this.source && this.context) {
      const effective = this.needleDown ? rate : 0;
      this.source.playbackRate.setTargetAtTime(Math.max(effective, 0.0001), this.context.currentTime, 0.02);
    }
    this.applyGain();
  }

  private applyGain(): void {
    if (!this.gain || !this.context) return;
    const audible = this.needleDown && this.rate > 0.02;
    this.gain.gain.setTargetAtTime(audible ? 1 : 0, this.context.currentTime, 0.04);
  }

  /** Advance the groove position; returns progress 0..1 (or 0 when disabled). */
  update(dt: number): number {
    if (!this.buffer) return 0;
    if (this.needleDown && this.rate > 0) {
      this.position = Math.min(this.duration, this.position + dt * this.rate);
      if (this.position >= this.duration && !this.endedFired) {
        this.endedFired = true;
        this.options.onEnded?.();
      }
    }
    return this.duration > 0 ? this.position / this.duration : 0;
  }

  /** Frequency magnitudes 0..255 for a visualizer; false when audio is not enabled. */
  spectrum(out: Uint8Array<ArrayBuffer>): boolean {
    if (!this.analyser) return false;
    this.analyser.getByteFrequencyData(out);
    return true;
  }

  dispose(): void {
    this.source?.stop();
    void this.context?.close();
  }
}
