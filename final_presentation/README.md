# Final Presentation logs

Logs of creating the final presentation

* Multirotor with Vnom 0 on NPFG not working (awkward)
  * `ffmpeg -ss 3.7 -t 5 -i MC_VelNom0_NPFG_Demo.mkv -filter_complex "[0:v] fps=30,scale=w=480:h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" mc_velnom0.gif`
* Vpath = 0 case
  * `ffmpeg -ss 4.1 -t 7.5 -i Vpath0_Demo.mkv -filter_complex "[0:v] fps=30,scale=w=480:h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" vpath0.gif`
* Vpath = 12 case
  * `ffmpeg -ss 4.0 -t 11 -i Vpath12_Demo.mkv -filter_complex "[0:v] fps=30,scale=w=480:h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" vpath12.gif`
* Multirotor vs Fixed Wing case
  * `ffmpeg -ss 3.7 -t 16 -i MC_FW_Compare_v3.mkv -filter_complex "[0:v] fps=30,scale=w=480:h=-1,split [a][b];[a] palettegen=stats_mode=single [p];[b][p] paletteuse=new=1" mc_fw_compare.gif`
* 