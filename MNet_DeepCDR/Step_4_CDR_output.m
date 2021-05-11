% You'll need to extract 'mat_scr.zip' for this to work.
clear; close all;
addpath(genpath(['mat_scr' filesep]));

raw_result = 'result_refuge';
img_list = dir([raw_result filesep ' *.png']);

img_num = size(img_list, 1);

for idx = 1 : img_num
img_name = img_list(idx).name;
img_map = imread([raw_result filesep img_name]);
[img_h, img_w, img_c] = size(img_map);

Disc_map = fun_Ell_Fit(img_map > 100, img_h, img_w, 1);
Cup_map = fun_Ell_Fit(img_map > 200, img_h, img_w, 1);
CDR_value = fun_CalCDR(Disc_map.fit_map, Cup_map.fit_map);

Seg_map = Disc_map.fit_map + Cup_map.fit_map;
Seg_map(Seg_map == 0) = 255;
Seg_map(Seg_map == 1) = 128;
Seg_map(Seg_map == 2) = 0;

save(['final_result' filesep img_name(1 : end - 4) '.mat'], 'CDR_value');
imwrite(uint8(Seg_map), ['final_result' filesep img_name(1 : end - 4) '.bmp']);
end
