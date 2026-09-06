// Catalogue of models the viewer knows how to present.
export interface Credit {
  author: string;
  source: string;
  license: string;
  url: string;
}

export interface AttachmentSpec {
  id: string;
  url: string;
  /** Lighter GLB served to phones (decimated mesh, small WebP textures). */
  mobileUrl?: string;
  credit: Credit;
  /** Axis the attachment's file treats as "up". Defaults to "y". */
  upAxis?: "y" | "z";
  /** Use raw mesh space, dropping node transforms (Sketchfab bakes a presentation tilt into the root node). */
  ignoreNodeTransforms?: boolean;
  /** Rescale so the attachment's footprint (longest horizontal side) is this many meters. */
  fitDiameter?: number;
  /** Center it on the detected platter and rest it on top of it. */
  placeOnPlatter: boolean;
  /** Extra vertical gap in meters (e.g. slipmat compression). */
  lift?: number;
  /** Rotate with the platter. */
  spin: boolean;
}

export interface TonearmSpec {
  /** Mesh indices (main GLB) that swing horizontally with the arm. */
  meshes: number[];
  /** Horizontal bearing of the arm, in the main file's scene space (after node transforms). */
  pivot: [number, number, number];
  /** Stylus tip at rest, in the main file's scene space. */
  stylus: [number, number, number];
  /** Distance from the platter axis where the stylus should sit, in meters. */
  playRadius: number;
  /** Fixed meshes (arm-rest clip and friends) that only make sense while the arm is parked. */
  hideAwayFromRest?: number[];
  /** Up to two scene-space boxes (meters) of fixed detail that is always hidden (stray indicator marks). */
  hideRegions?: Array<{ min: [number, number, number]; max: [number, number, number] }>;
  /**
   * Meshes that fuse arm parts with unrelated chassis parts: triangles inside the
   * scene-space box are split off into their own mesh and move with the arm.
   */
  splitMeshes?: Array<{ mesh: number; min: [number, number, number]; max: [number, number, number] }>;
}

export type DeckButtonId = "startStop" | "speed33" | "speed45" | "quartz";

export interface DeckControlsSpec {
  /**
   * Physical buttons as scene-space boxes (meters). Tapping one triggers its action and
   * the geometry inside the box dips (press animation); LEDs inside follow the button state.
   */
  buttons?: Array<{ id: DeckButtonId; min: [number, number, number]; max: [number, number, number] }>;
  /**
   * Pitch fader: knob meshes, travel direction in scene space (positive = faster),
   * half travel in meters and the speed change at full travel (fraction).
   */
  pitch?: {
    /** Meshes that travel with the fader knob. */
    meshes: number[];
    /** Extra meshes that also start a drag (the slot/track). */
    grabMeshes?: number[];
    axis: [number, number, number];
    halfTravel: number;
    maxPitch: number;
  };
}

export interface AudioSpec {
  url: string;
  title: string;
  artist: string;
  /** Stylus radius at the first and last groove, meters; the arm creeps between them. */
  leadInRadius: number;
  runOutRadius: number;
}

export interface LabelSpec {
  /** Square (or center-cropped) image painted on the record label. */
  image: string;
  /** Label diameter in meters (a 12" record label is 100 mm). */
  diameter: number;
  /** Spindle hole diameter in meters. */
  hole: number;
}

export interface ModelSpec {
  id: string;
  /** Display name shown in the HUD. */
  title: string;
  subtitle: string;
  url: string;
  /** Lighter GLB served to phones (decimated mesh, small WebP textures). */
  mobileUrl?: string;
  credit: Credit;
  /** Nodes whose names match rotate around the vertical axis when "spin" is on. */
  spinPattern?: RegExp;
  /**
   * Detect the platter geometrically: thin meshes with a near-square footprint that
   * spans at least this fraction of the model's footprint. Used when nodes are unnamed.
   */
  detectPlatter?: { minFootprint: number };
  /** Axis the source file treats as "up". Defaults to "y". */
  upAxis?: "y" | "z";
  /** Extra rotation around the vertical axis (radians) so the model faces the camera. */
  rotateY?: number;
  /** Longest side of the model in viewer units. */
  targetSize: number;
  /** Enable the gold -> black "Negro" finish recolor for this model. */
  recolorable: boolean;
  /** Materials that are indicator LEDs: lit only while the motor runs. */
  ledMaterialPattern?: RegExp;
  /** Initial camera pose. */
  camera: { yaw: number; pitch: number; distance: number };
  attachments?: AttachmentSpec[];
  tonearm?: TonearmSpec;
  controls?: DeckControlsSpec;
  audio?: AudioSpec;
  label?: LabelSpec;
}

export const MODELS: Record<string, ModelSpec> = {
  turntable: {
    id: "turntable",
    title: "Audio-Technica AT-LP120XUSB",
    subtitle: "Professional direct-drive turntable · Black",
    url: "/models/turntable.glb",
    mobileUrl: "/models/turntable.mobile.glb",
    credit: {
      author: "Mateusz Kołakowski",
      source: "Pioneer PLX-1000 Turntable Ltd. (Remastered)",
      license: "CC BY 4.0",
      url: "https://sketchfab.com/3d-models/pioneer-plx-1000-turntable-ltd-remastered-147cd49228e24746a6a2b3c45440d989",
    },
    spinPattern: /platter|plate|vinyl|record|disc|talerz|plyta|płyta/i,
    detectPlatter: { minFootprint: 0.55 },
    targetSize: 1.25,
    recolorable: true,
    ledMaterialPattern: /emission|led/i,
    camera: { yaw: -0.65, pitch: 0.5, distance: 2.9 },
    attachments: [
      {
        id: "vinyl",
        url: "/models/vinyl.glb",
        mobileUrl: "/models/vinyl.mobile.glb",
        credit: {
          author: "AleixoAlonso",
          source: '12" Vinyl Record',
          license: "CC BY 4.0",
          url: "https://sketchfab.com/3d-models/12-vinyl-record-dd27284e8c1a4622b3a0433c7cc8ee72",
        },
        ignoreNodeTransforms: true,
        upAxis: "z",
        fitDiameter: 0.3,
        placeOnPlatter: true,
        lift: 0.0004,
        spin: true,
      },
    ],
    // Meshes below are the headshell lead wires (0-3), bearing housing (12), cartridge
    // (17, 18), counterweight (19), headshell (21), arm tubes (23, 35) and finger-lift
    // collar (24); they swing together around the bearing. The arm lift and cue lever stay.
    // The GLB root node maps the authoring Z-up axes to Y-up: scene = (x, z, -y).
    tonearm: {
      meshes: [0, 1, 2, 3, 12, 17, 18, 19, 21, 23, 24, 35],
      hideRegions: [],
      // The cartridge body shares a mesh with other plastic bits; carve it out so it rides the arm.
      splitMeshes: [
        { mesh: 34, min: [0.09, 0.097, 0.118], max: [0.142, 0.145, 0.168] },   // headshell clip
        { mesh: 32, min: [0.112, 0.104, 0.10], max: [0.148, 0.126, 0.132] },   // cartridge body (fused with cue lever/lift bar)
      ],
      pivot: [0.152, 0.12, -0.085],
      // Lowest point of the moving assembly (headshell underside / lead wires), so the
      // tilt never pushes geometry through the record.
      stylus: [0.115, 0.108, 0.138],
      playRadius: 0.146,
    },
    audio: {
      url: "/audio/track.m4a",
      title: "10:15 Saturday Night",
      artist: "The Cure",
      leadInRadius: 0.146,
      runOutRadius: 0.06,
    },
    label: { image: "/audio/cover.jpg", diameter: 0.1, hole: 0.0075 },
    controls: {
      buttons: [
        { id: "startStop", min: [-0.218, 0.084, 0.128], max: [-0.1705, 0.1, 0.172] }, // round START/STOP + ring, front-left
        { id: "speed33", min: [-0.1775, 0.084, 0.144], max: [-0.1500, 0.1, 0.172] },   // left bar (x -0.175..-0.151)
        { id: "speed45", min: [-0.1500, 0.084, 0.144], max: [-0.1225, 0.1, 0.172] },   // right bar (x -0.149..-0.125)
        { id: "quartz", min: [0.138, 0.086, 0.076], max: [0.192, 0.112, 0.11] },    // pitch lock by the fader (+ its LED)
      ],
      // Fader knob rides the slot on the right edge; toward the front = faster.
      pitch: { meshes: [26], grabMeshes: [10], axis: [0, 0, 1], halfTravel: 0.055, maxPitch: 0.08 },
    },
  },
  helmet: {
    id: "helmet",
    title: "Damaged Helmet",
    subtitle: "Pipeline smoke test · Khronos glTF sample asset",
    url: "/models/DamagedHelmet.glb",
    credit: {
      author: "theblueturtle_",
      source: "KhronosGroup/glTF-Sample-Assets",
      license: "CC BY-NC 4.0",
      url: "https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DamagedHelmet",
    },
    targetSize: 1.0,
    recolorable: false,
    camera: { yaw: 0.4, pitch: 0.2, distance: 3.4 },
  },
};

export const DEFAULT_MODEL_ID = "turntable";

export function resolveModel(id: string | null | undefined): ModelSpec {
  return (id && MODELS[id]) || MODELS[DEFAULT_MODEL_ID];
}
