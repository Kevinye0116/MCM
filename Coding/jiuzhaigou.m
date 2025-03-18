% Create data vectors
x = [1474284; 2253229; 1129669; 4346604; 5824519]; % Population data
y = [46.52; 57.49; 45.23; 72.14; 88.35]; % Revenue data

% Calculate linear fit
p = polyfit(x, y, 1);
yfit = polyval(p, x);

% Create scatter plot and fitting line
figure;
plot(x, y, 'bo', 'MarkerSize', 8); % Plot original data points
hold on;
plot(x, yfit, 'r-', 'LineWidth', 2); % Plot fitting line
grid on;

% Add labels and title
xlabel('Population');
ylabel('Revenue');
title('Linear Relationship between Population and Revenue');
legend('Original Data', 'Linear Fit');

% Display fitting equation on the plot
equation = sprintf('y = %.4fx + %.2f', p(1), p(2));
text(min(x), max(y), equation, 'FontSize', 12);

% Optimize graph display
set(gca, 'FontSize', 12);
box on;