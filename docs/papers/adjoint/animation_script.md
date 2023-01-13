demo animations脚本，实际结果稍有不同。见assets branch里张宇澄制作的
`assets/{time_evolution,pmwd_optim}.mp4`

1. 时间反演
  (a) 向前模拟，演化时间为a（宇宙大小），初始a=1/64，结束a=1，
      在角落里标记当前时间"Universe scale factor a = 0.xxx"
    * a=1/64时，camera在近处。观测到一颗颗粒子在立方格点上，粒子弥散较小以观测其粒子性
    * 随着演化，camera移到远处，同时绕盒子的长轴转动2圈，粒子弥散逐渐变为视觉最佳值
    * 字幕
      - Dark matter forms nonlinear structures from nearly uniform
        initial conditions as the Universe evolves.
      - pmwd is a differentiable and memory efficient code to simulate
        the large-scale structure of the Universe.
        (“pmwd”请用附带的NovaRoundSlim-BookOblique.ttf字体，其它请用Palatino)
    * a=1时，camera挪到远处
  (b) 向后模拟，时间反演，初始a=1，结束a=1/64，角落标记时间
    * a=1时，camera在远处，短暂停顿
    * 随着演化，camera移到近处，同时绕盒子的长轴转动2圈(与以上同向)，粒子弥散逐渐变小
    * 字幕
      - Reversing the arrow of time, the evolution history can be
        reconstructed,
      - so gradients can be propagated backward at the same time without
        saving the whole history during the forward run.
    * a=1/64时，camera在近处。观测到粒子回到了立方格点上，粒子弥散较小以观测其粒子性，短暂停顿
  (c) 向前模拟，数据同1a，快进2x速度，初始a=1/64，结束a=1，角落标记时间
    * a=1/64时，camera在近处。观测到一颗颗粒子在立方格点上
    * 随着演化，camera移到远处，无转动，粒子弥散逐渐变为视觉最佳值
    * 字幕
      - The gradients can help optimizing certain objectives.
    * a=1时，camera挪到远处
2. 梯度下降优化
  * 演化“时间”为优化步数i(的对数)，初始i=1, 结束i=10000，
    在角落里标记当前时间"optimization iteration i = yyy"
  * camera在远处，小fov
  * camera绕盒子的长轴转动4圈
    与1中的匀速转动不同，为了展示优化过程中在盒面方向的投影，
    当平行盒子长轴的4个面面向camera时转速较慢，而转离时转速较快，
    可以采用类似a + 0.5 b ( 1 - cos 4θ )的形式，
    例如当b=2a是的最慢与最快转速比为1:3
  * 粒子弥散设为视觉最佳值
  * 转第1圈时，i从1变到10; 转第2圈时，i从10变到100; 转第3圈时，i从100变到1000; 转第4圈时，i从1000变到10000；之后短暂停顿
  * 字幕(一段时间后消失)
    - In this toy example, we update the initial conditions of 16x27x16
      dark matter particles iteratively to form some interesting
      patterns.
