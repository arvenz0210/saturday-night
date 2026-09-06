# vinyl · AT-LP120XUSB WebGPU viewer

Visor 3D en tiempo real de una bandeja giradisco profesional (Audio-Technica
AT-LP120XUSB, acabado negro) construido con **Next.js 16** y **[vgpu](https://vgpu.sh)**,
la librería WebGPU de Vercel Labs. Todo el render corre en WebGPU a través de la API
de vgpu (`draw`, `effect`, `target`, `frameLoop`, `uniforms`, módulos WGSL con
`import`/`export`), sin three.js.

## Qué renderiza

- **PBR metálico‑rugoso** con normal mapping, oclusión ambiental y emisivo, leído
  directamente del GLB por un parser propio (`lib/gltf/glb.ts`, sin dependencias).
- **Iluminación de estudio basada en imagen**: un entorno HDR procedural con tres
  softboxes (key, fill, rim) se hornea al arrancar en un mapa especular prefiltrado
  GGX (7 niveles de rugosidad), un mapa de irradiancia y una LUT BRDF split‑sum.
- **Sombra suave** desde la luz principal (shadow map 2048², 16 taps Poisson rotados).
- **HDR + MSAA 4×**, bloom (threshold + gaussiano separable) y tone mapping ACES.
- **Cámara orbital** de vgpu (`orbitControls`) con auto‑rotación, plato girando a 33⅓ rpm,
  y un acabado "Negro" que recolorea el dorado del modelo original a negro satinado.

## Ejecutar

```bash
pnpm install
pnpm dev
# http://localhost:3000            -> bandeja giradisco
# http://localhost:3000/?model=helmet -> Damaged Helmet (Khronos), prueba de pipeline
```

Requiere un navegador con WebGPU (Chrome/Edge 113+, Safari 26+).

## Modelos

Los `.glb` viven en `public/models/` y están fuera de git por tamaño.

| id          | archivo                              | fuente                                                                                                                                  |
| ----------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `turntable` | `public/models/turntable.glb`        | [Pioneer PLX‑1000 Turntable Ltd. (Remastered)](https://sketchfab.com/3d-models/pioneer-plx-1000-turntable-ltd-remastered-147cd49228e24746a6a2b3c45440d989) · Mateusz Kołakowski · CC BY 4.0 |
| `helmet`    | `public/models/DamagedHelmet.glb`    | [Khronos glTF Sample Assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/DamagedHelmet) · CC BY‑NC 4.0            |

No existe un modelo libre del AT‑LP120XUSB; el PLX‑1000 comparte la misma
arquitectura (clon del Technics SL‑1200: brazo en S, plato con puntos de estrobo,
pitch fader) y el acabado "Negro" lo acerca al AT‑LP120X negro. Para usar otro GLB,
añade una entrada en `lib/models.ts`.

## Estructura

```
app/                      layout + página (Server Component) que monta el visor
components/               TurntableViewer.tsx (client) + HUD
lib/gltf/glb.ts           parser GLB -> buffers intercalados, materiales, instancias
lib/render/engine.ts      ciclo de render vgpu: sombra -> PBR HDR -> bloom -> post
lib/render/environment.ts horneado IBL (entorno, prefiltro GGX, irradiancia, LUT)
lib/render/textures.ts    subida de imágenes + mipmaps, blit, samplers glTF
lib/render/shaders/*.wgsl módulos WGSL (common, pbr, shadow, env, prefilter, ...)
```

Atajos: arrastrar orbita, rueda acerca, `R` reinicia la cámara, `H` oculta la UI.

## Disco y brazo

`lib/models.ts` describe la escena además del GLB principal:

- **`attachments`**: el disco (`public/models/vinyl.glb`) se recoloca sobre el plato
  detectado geométricamente (mallas finas de huella casi cuadrada), se escala a su
  diámetro real (`fitDiameter: 0.3` m) y gira con él a 33⅓ rpm. Los GLB de Sketchfab
  traen una inclinación de presentación en el nodo raíz; `ignoreNodeTransforms` usa el
  espacio de malla en bruto y `upAxis` lo endereza.
- **`tonearm`**: lista de mallas que forman el brazo (tubo, headshell, cápsula, cables,
  contrapeso). El visor lo gira sobre su eje hasta que la púa queda a `playRadius`
  (146 mm, surco de entrada) y lo inclina hasta apoyarla en la superficie del disco.
  La geometría de la púa es la original del modelo, sin simplificar.

Los `console.info` `[viewer]` resumen tiempos de carga y el ángulo aplicado al brazo.

## Gestos y audio

- **Brazo**: toca el headshell/brazo para levantarlo y llevarlo al reposo, o para
  bajarlo sobre el surco de entrada (secuencia: levanta, gira, baja). Mientras suena,
  el brazo avanza lentamente hacia el final del disco.
- **START/STOP**: toca el botón redondo del frente. El plato acelera y frena con inercia,
  y el audio sigue esa velocidad (arranque y frenada audibles).
- **Pitch**: arrastra la corredera del lado derecho (±8 %). Cambia la velocidad del plato
  y el `playbackRate` del audio; el HUD muestra % y rpm.
- **Audio**: `public/audio/track.m4a` + `public/audio/cover.jpg` (carátula pintada en la
  etiqueta del disco). Ambos están fuera de git y del deploy público: son material con
  derechos, descargado para uso local con `yt-dlp` desde el enlace indicado en
  `lib/models.ts`. Sin ellos el visor funciona igual (sin sonido y con la etiqueta original). El sonido se
  activa con el botón "Activar sonido" (política de autoplay) y solo suena con la púa
  apoyada; el espectro del HUD sale de un `AnalyserNode`.

El picking usa las cajas envolventes de cada malla (rayo desde la cámara) con prioridad
para las piezas interactivas; el arrastre del pitch captura el puntero y no orbita la cámara.

## Vista "mesa" (por defecto) y estudio

La página usa siempre la vista "mesa" (añade `?layout=studio` para el set oscuro con
cámara orbital y panel de controles): fondo blanco, cámara fija
picada (~54°) ajustada al ancho, sombra suave (PCF ancho) compuesta en post sobre el blanco,
y debajo solo la carátula y el título. No hay botones: el primer toque activa el audio y los
gestos sobre la escena (brazo, START/STOP, pitch) siguen activos. La cámara se mueve con un
dedo sobre la mesa (fuera de la bandeja) o con dos dedos (pinza = zoom, giro = yaw).
El reposabrazos se oculta mientras el brazo está sobre el disco, y los LED azules solo se
encienden con el motor en marcha.

## Botones físicos

START/STOP, 33, 45 y QUARTZ son cajas en el espacio de la escena (`controls.buttons` en
`lib/models.ts`). Al tocarlas la geometría dentro de la caja se hunde ~1 mm (animación de
pulsación en el vertex shader) y sus LED siguen el estado: 33/45 seleccionan la velocidad
nominal del plato (el audio sube a 45 rpm como en un disco real), QUARTZ bloquea el pitch
a 0 % (LED encendido) hasta volver a pulsarlo. El brazo avanza hacia el centro al ritmo del
plato como en una cara real (~20 min de surco).
