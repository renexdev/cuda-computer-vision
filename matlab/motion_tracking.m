close all; clear all;
edges_1 = edge_detect('lobby_1.jpg', 25, 35, 5);
edges_2 = edge_detect('lobby_2.jpg', 25, 35, 5);
figure, imshow(edges_1);
figure, imshow(edges_2);

height = 2988;
width = 5312;

difference = zeros(height, width);
apron_width = 2;
for x = 1:width
    for y = 1:height
        if edges_1(y, x) ~= edges_2(y, x)
            difference(y, x) = 1;
            for x_apron = -apron_width:apron_width
                for y_apron = -apron_width:apron_width
                    if x + x_apron > 0 && y + y_apron > 0 && x + x_apron < width && y + y_apron < height && edges_1(y + y_apron, x + x_apron) == edges_2(y, x)
                        difference(y, x) = 0;
                    end
                end
            end
        end
    end
end
figure, imshow(difference);

horizontal_divisions = 10;
vertical_divisions = 6;
area_coverage = zeros(vertical_divisions, horizontal_divisions);
horizontal_box_size = floor(width/horizontal_divisions);
vertical_box_size = floor(height/vertical_divisions);
box_size = horizontal_box_size * vertical_box_size;
for box_x_index = 1:floor(width/horizontal_box_size)
    for box_y_index = 1:floor(height/vertical_box_size)
        num_differences = 0;
        for x = (box_x_index - 1)*horizontal_box_size:box_x_index*horizontal_box_size
            for y = (box_y_index - 1)*vertical_box_size:box_y_index*vertical_box_size
                if x > 0 && y > 0 && x < width && y < height && difference(y, x) == 1
                    num_differences = num_differences + 1;
                end
            end
        end
        area_coverage(box_y_index, box_x_index) = num_differences/box_size;
    end
end
figure, mesh(area_coverage);

maxima = [];
for box_x_index = 1:horizontal_divisions
    for box_y_index = 1:vertical_divisions
        if area_coverage(box_y_index, box_x_index) > 0.01
            maxima = [[box_x_index; box_y_index], maxima];
        end
    end
end
box = zeros(height, width);
[null, num_maxima] = size(maxima);
for i = 1:num_maxima
    for x = (maxima(1, i) - 1)*horizontal_box_size:maxima(1, i)*horizontal_box_size
        for y = (maxima(2, i) - 1)*vertical_box_size:maxima(2, i)*vertical_box_size
            box(y, x) = 1;
        end
    end
end
figure, imshow(box);