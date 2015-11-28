% close all;
% clear all;
function edges = edge_detect(filename, low_threshold, high_threshold, sigma)
    image = imread(filename);
    image_bw = double(rgb2gray(image));
    [height, width] = size(image_bw);

    k = 4;
    h = [];
    for i = 1:2*k + 1
        for j = 1:2*k + 1
            h(i, j) = 1/(2*pi*sigma^2)*exp(-((i - k - 1)^2 + (j - k - 1)^2)/(2*sigma^2));
        end
    end
    h = h/sum(sum(h));
    filtered = conv2(h, image_bw);
%     figure, mesh(image_bw);
%     figure, mesh(filtered);
%     figure, imshow(image);

    % sobel
    % Gx = conv2([-1 0 1; -2 0 2; -1 0 1], filtered);
    % Gy = conv2([-1 -2 -1; 0 0 0; 1 2 1], filtered);
    % implemented separably
    Gx = conv2([1; 2; 1], conv2([1 0 -1], filtered));
    Gy = conv2([1; 0; -1], conv2([1 2 1], filtered));
    G = sqrt(Gx.^2 + Gy.^2);
    theta = atan2(Gy, Gx);
    edges = zeros(height, width);
    edges_no_suppression = zeros(height, width);
    for y = 3:height - 2
        for x = 3:width - 2
            vertical_check = (pi/3 < theta(y, x) && theta(y, x) < 2*pi/3) || (-2*pi/3 < theta(y, x) && theta(y, x) < -pi/3);
            is_vertical_max = G(y, x) > G(y + 1, x) && G(y, x) > G(y - 1, x);
            horizontal_check = (-pi/6 < theta(y, x) && theta(y, x) < pi/6) || (-pi < theta(y, x) && theta(y, x) < -5*pi/6) || (5*pi/6 < theta(y, x) && theta(y, x) < pi);
            is_horizontal_max = G(y, x) > G(y, x + 1) && G(y, x) > G(y, x - 1);
            diagonal_check = ~vertical_check && ~horizontal_check;
            is_diagonal_max = G(y, x) > G(y + 1, x + 1) && G(y, x) > G(y - 1, x - 1);
            threshold_condition = G(y, x) > high_threshold;
            if (((vertical_check && is_vertical_max) || (horizontal_check && is_horizontal_max) || (diagonal_check && is_diagonal_max)) && threshold_condition)
                for i = -2:2
                    for j = -2:2
                        if (G(y + i, x + j) > low_threshold)
                            edges(y, x) = 1;
                            edges(y + i, x + j) = 1;
                        end
                    end
                end
            end
            if (threshold_condition)
                edges_no_suppression(y, x) = 1;
            end
        end
    end
%     figure, imshow(edges);
%     figure, imshow(edges_no_suppression);
end
