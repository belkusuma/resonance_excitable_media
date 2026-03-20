%% Solving the FHG with the forward Euler method
%% Legacy code to replicate (M. Perc, ‘Spatial coherence resonance in excitable media’, 
%% Phys. Rev. E, vol. 72, no. 1, p. 016207, Jul. 2005, doi: 10.1103/PhysRevE.72.016207.)
%% Done by Iris Marmouset-de la Taille, Nov 2023


% Parameters
m = 100;% x axis 
n = 100; % y axis
delta_x = 1; delta_y = 1; delta_t = 0.1;
t = 0:15;% range of time of the simulation
timepoints = t(1):delta_t:t(end); % vector of timepoints 
a = 0.75; % float, FHN parameter (else: 1.05)
b = 0.01; % float, FHN parameter 
epsilon = 0.05 ; % float, allows the local dynamics of u being much faster than that of v 
sigma = 0.24; % float, σ of the white Gaussian noise ξ -> allows coherence resonnance
D = 1.2; % diffusion coefficient (uniform in space at the moment)



% Create the u-mesh and v-mesh :
u = zeros(m, n, width(timepoints)); % membrane potential u(x,y,t)
v = zeros(m, n, width(timepoints)); % conductance of potassium channels v(x,y,t)


% Equations: 
% ∂u/∂t = f(u, v) + D * ∇²u + noise;  
% ∂v/∂t - g(u, v) = 0;
% f_uv = (1/eps)*u*(1-u)*(u-(v + b)/a);
% f(u,v) = piecewise(u <= 1, f_uv , u > 1, -abs(f_uv));
% g(u,v) = piecewise(v >= 0, u-v , v < 0, abs(u-v));

% ∂u(x,y,t)/∂t = (u(x,y,t+delta_t) - (u(x,y,t)) / delta_t
% => u(x,y,t+1) = (f(u,v) + D*laplacian_u + noise) * delta_t + u(x,y,t)

% ∇²u(x,y,t) = (u(x+delta_x,y,t) + (u(x-delta_x,y,t) - 2*u(x,y,t)) / delta_x^2 
%            + (u(x,y+delta_y,t) + (u(x,y-delta_y,t) - 2*u(x,y,t)) / delta_y^2


% For the moment I store all the values along time because I wanted to have a
% look at the videos, but later we could avoid storing in a 3D tensor 

for t=1:width(timepoints) % for each timepoint i.e for each plan (x,y) of my 3D tensor(x,y,t)
    for x=2:m-1 % for each row of my mesh % 
        for y=2:n-1 %for each column of my mesh % 
            
            f = (1/epsilon) * u(x,y,t) * (1-u(x,y,t)) * (u(x,y,t) - ((v(x,y,t)+b)/a));
            f_uv = (f*(u(x,y,t)<=1)) + (-abs(f)*(u(x,y,t)>1)) ; 

            u_xx = (u(x+1,y,t)+u(x-1,y,t)-2*u(x,y,t)) / delta_x^2;
            u_yy = (u(x,y+1,t)+u(x,y-1,t)-2*u(x,y,t)) / delta_y^2;
            laplacian_u = u_xx + u_yy; 

            noise = sigma*randn; % this is the Gaussian noise that is tuned 

            g = (u(x,y,t) - v(x,y,t));
            g_uv = (g *(v(x,y,t)>= 0)) + (abs(g)* (v(x,y,t) < 0));

            u(x,y,t+1) = (f_uv + D*laplacian_u +noise) * delta_t + u(x,y,t);
            v(x,y,t+1) = (g_uv) * delta_t + v(x,y,t);

            % Neumann boundary conditions: derivative on the normal direction is fixed (here to 0)
            % i.e u_x(x=1,y,t) = u_x(x=m,y,t) = u_y(x,y=1,t) = u_y(x,y=n,t) = 0
            if x == 2
                % Left boundary: x = 1
                u(x-1,y,t+1) = u(x,y,t+1);
                v(x-1,y,t+1) = v(x,y,t+1); 
            end
            if x == (m-1)
                % Right boundary: x = m
                u(x+1,y,t+1) = u(x,y,t+1);
                v(x+1,y,t+1) = v(x,y,t+1);
            end
            if y == 2
                % Top boundary: y = 1
                u(x,y-1,t+1) = u(x,y,t+1);
                v(x,y-1,t+1) = v(x,y,t+1);
            end
            if y == (n-1)
                % Bottom boundary: y = n
                u(x,y+1,t+1) = u(x,y,t+1);
                v(x,y+1,t+1) = v(x,y,t+1);
            end

            clear f_uv laplacian_u noise g_uv g f u_xx u_yy 
        end        
    end
end

clear m n delta_y delta_x delta_t t timepoints a b epsilon D sigma x y t 



%% Show the Movie of the simulation

for t=1:size(u,3)
    u_t = u(:,:,t);
    imshow(u_t, [min(u(:)), max(u(:))], 'Colormap', jet);
    % colorbar;
    title(['Time Step: ', num2str(t)]);
    drawnow;
end

clear t u_t fig  


%% Storing the video 

outputVideo = VideoWriter('m224_n224_s024_D120.mp4', 'MPEG-4');
outputVideo.FrameRate = 30; 
open(outputVideo);


for t=1:size(u,3)
    u_t = u(:,:,t);
    figure('Visible', 'off');
    imshow(u_t, [min(u(:)), max(u(:))], 'Colormap', jet);
    % colorbar;
    title(['Time Step: ', num2str(t)]);
    drawnow; 
    
    frame = getframe;  % getting the current frame as an image
    writeVideo(outputVideo, frame); % writting the frame to the video
    
    % Close the figure to release resources
    close(gcf);
end

% Close the video file
close(outputVideo);

clear t u_t fig  outputVideo frame 

%% Show a specific frame of the simulation 

timepoint = 140;
imagesc(u(:,:,timepoint))

colormap('jet'); 
colorbar; 
title(['Membrane potential u at timepoint ' num2str(timepoint)]);
