// Video manifest. This is the only file you need to edit to add results.
//
// - Put the .mp4 under static/videos/<folder>/ and set `src` to its path.
// - Optional `aspect: "W / H"` if a clip is not 1280x576.
// - Keep filenames free of spaces, usernames, dates of real people, lab names.
// - Clips are grouped by object id (pusht / rope / motherboard). An object with
//   no clips in a section gets no button there.
//
// Layout of each clip: top row = ground truth, bottom row = ACTS prediction
// (comparison clips: one row per method under the ground truth);
// columns = scene, wrist L, wrist R, tactile L, tactile R.
// Short clips: 8 input frames + 8 predicted frames. Long clips: 8 input + 120 predicted.

window.ACTS_VIDEOS = {
  tasks: [
    { id: "pusht", name: "PushT" },
    { id: "rope", name: "Rope" },
    { id: "motherboard", name: "Motherboard" },
  ],

  shortHorizon: {
    pusht: [
      { src: "static/videos/short/pusht_e019_01.mp4" },
      { src: "static/videos/short/pusht_e019_02.mp4" },
      { src: "static/videos/short/pusht_e019_03.mp4" },
      { src: "static/videos/short/pusht_e019_21.mp4" },
      { src: "static/videos/short/pusht_e019_25.mp4" },
    ],
    rope: [
      { src: "static/videos/short/rope_e019_01.mp4" },
      { src: "static/videos/short/rope_e019_02.mp4" },
      { src: "static/videos/short/rope_e019_05.mp4" },
      { src: "static/videos/short/rope_e019_14.mp4" },
      { src: "static/videos/short/rope_e019_15.mp4" },
    ],
    motherboard: [
      { src: "static/videos/short/motherboard_e019_02.mp4" },
      { src: "static/videos/short/motherboard_e019_07.mp4" },
      { src: "static/videos/short/motherboard_e019_11.mp4" },
      { src: "static/videos/short/motherboard_e019_14.mp4" },
      { src: "static/videos/short/motherboard_e019_15.mp4" },
    ],
  },

  // Human-trained model evaluated on robot episodes, no robot fine-tuning.
  robot: {
    pusht: [
      { src: "static/videos/robot/pusht_robot2_01047.mp4" },
      { src: "static/videos/robot/pusht_robot3_00343.mp4" },
      { src: "static/videos/robot/pusht_robot2_00991.mp4" },
    ],
    rope: [
      { src: "static/videos/robot/rope_robot4_00950.mp4" },
      { src: "static/videos/robot/rope_robot4_01050.mp4" },
      { src: "static/videos/robot/rope_robot4_01591.mp4" },
    ],
    motherboard: [
      { src: "static/videos/robot/motherboard_robot2_00190.mp4" },
      { src: "static/videos/robot/motherboard_robot1_00695.mp4" },
      { src: "static/videos/robot/motherboard_robot2_01872.mp4" },
    ],
  },

  // Force-aware vs. pose-only actions on the same test window, stacked under one ground-truth row.
  // `force`: recorded normal force (N) per frame, 16 frames at 15 fps, 8 input + 8 predicted.
  ablations: {
    pusht: [
      {
        label: "Making contact",
        src: "static/videos/ablation/pusht_12.mp4",
        rows: ["Force-aware action (ACTS)", "Pose-only action"],
        force: {
          left: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          right: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.43, 0.77, 1.17, 1.17, 1.17, 2.64],
        },
      },
      {
        label: "Breaking contact",
        src: "static/videos/ablation/pusht_02.mp4",
        rows: ["Force-aware action (ACTS)", "Pose-only action"],
        force: {
          left: [2.82, 2.94, 2.94, 5.15, 5.15, 5.26, 5.26, 5.26, 5.15, 2.94, 1.17, 0, 0, 0, 0, 0],
          right: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        },
      },
    ],
    rope: [
      {
        label: "Making contact",
        src: "static/videos/ablation/rope_27.mp4",
        rows: ["Force-aware action (ACTS)", "Pose-only action"],
        force: {
          left: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, 0, 0],
          right: [0.01, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 3.92, 3.92, 4.73, 4.73, 4.73],
        },
      },
    ],
  },

  failures: {
    pusht: [
      { src: "static/videos/failure/pusht_e013_13_short.mp4", mode: "Wrong dynamics.", caption: "The predicted T position diverges from the ground truth in the scene and tactile views." },
      { src: "static/videos/failure/pusht_episode_010_seg02_t00208.mp4", mode: "Wrong dynamics.", caption: "The predicted T position diverges from the ground truth in the scene and tactile views, and the T deforms.", aspect: "1326 / 530" },
    ],
    rope: [
      { src: "static/videos/failure/rope_e019_04.mp4", mode: "Wrong contact.", caption: "The predicted contact location and tactile image are wrong." },
      { src: "static/videos/failure/rope_e019_13.mp4", mode: "Wrong contact.", caption: "Contact is predicted incorrectly in the last four frames." },
    ],
    motherboard: [
      { src: "static/videos/failure/motherboard_e019_01.mp4", mode: "Wrong tactile imprint.", caption: "Both tactile imprints are wrong; the motherboard's complex geometry is hard to predict." },
      { src: "static/videos/failure/motherboard_e019_05.mp4", mode: "Wrong imprint and distortion.", caption: "The left tactile imprint is wrong and the motherboard is distorted in the wrist views, owing to its complex geometry." },
    ],
  },

  longHorizon: {
    pusht: [
      { src: "static/videos/long/pusht_03_long.mp4" },
      { src: "static/videos/long/pusht_10_long.mp4" },
    ],
    rope: [
      { src: "static/videos/long/rope_03_long.mp4" },
      { src: "static/videos/long/rope_05_long.mp4" },
    ],
    motherboard: [
      { src: "static/videos/long/motherboard_custom_e010_01_long.mp4" },
      { src: "static/videos/long/motherboard_custom_e010_02_long.mp4" },
    ],
  },

  // ACTS vs. the VT-WM-style baseline on the same test window, stacked under one ground-truth row.
  baselines: {
    pusht: [
      {
        label: "Making contact",
        src: "static/videos/baselines/pusht_05.mp4",
        rows: ["ACTS", "VT-WM-style"],
      },
      {
        label: "Breaking contact",
        src: "static/videos/baselines/pusht_25.mp4",
        rows: ["ACTS", "VT-WM-style"],
      },
    ],
    rope: [
      {
        label: "Making contact",
        src: "static/videos/baselines/rope_31.mp4",
        rows: ["ACTS", "VT-WM-style"],
      },
      {
        label: "Breaking contact",
        src: "static/videos/baselines/rope_24.mp4",
        rows: ["ACTS", "VT-WM-style"],
      },
    ],
    motherboard: [
      {
        label: "Making contact",
        src: "static/videos/baselines/motherboard_05.mp4",
        rows: ["ACTS", "VT-WM-style"],
      },
      {
        label: "Breaking contact",
        src: "static/videos/baselines/motherboard_02.mp4",
        rows: ["ACTS", "VT-WM-style"],
      },
    ],
  },
};
