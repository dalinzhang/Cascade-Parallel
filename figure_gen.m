%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train profile plot
% export the figure as .eps format will keep the figure in high resolution in latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_name = "/home/dadafly/program/CNN_LSTM/result/conv_3l_win_10_108_fc_rnn2_fc_1024_N_075_train/conv_3l_win_10_108_rnn2_fc_1024_N.xlsx";
[data, header] = xlsread(file_name, 'result');

epochs = data(:,1);
test_accuracy = data(:,2);
test_loss = data(:,3);
train_accuracy = data(:,4);
train_loss = data(:,5);

figure;
hold on;

yyaxis left
plot(epochs(1:300), train_accuracy(1:300), 'DisplayName', 'train accuracy','LineWidth',3);
plot(epochs(1:300), test_accuracy(1:300), 'DisplayName', 'test accuracy','LineWidth',3);

set(gca,'FontSize',24,'FontWeight','bold','linewidth',2); % set axes style

xlabel('Epochs','FontSize',24,'FontWeight','bold');
ylabel('Accuracy','FontSize',24,'FontWeight','bold');

box on
grid on
% grid minor

yyaxis right
plot(epochs(1:300), train_loss(1:300), 'DisplayName', 'train loss','LineWidth',3);
plot(epochs(1:300), test_loss(1:300), 'DisplayName', 'test loss','LineWidth',3);

ylabel('Loss','FontSize',24,'FontWeight','bold');
lgd =legend('show');
lgd.Box = 'on';
%lgd.LineWidth = 1.5;
%lgd.FontSize = 18;
%lgd.FontWeight = 'normal';

hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% test_error/train time vs training proportion
% export the figure as .eps format will keep the figure in high resolution in latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

file_name = "/home/dadafly/program/CNN_LSTM/result/train_proportion_acc_.xlsx";
[data, header] = xlsread(file_name);

train_proportion 	= data(:,1);
test_accuracy 		= data(:,2);
train_time 			= data(:,8);

figure;
hold on;

yyaxis right 
h = plot(train_proportion, test_accuracy ,'-d', 'DisplayName', 'validation accuracy','LineWidth',3);

% set with plot display in front or backgroud
ax = gca;
ax.SortMethod = 'depth';
%********************************************

set(gca, 'FontSize',24,'FontWeight','bold','linewidth',2); % set axes style
 
xlabel('Train Proportion (%)','FontSize',24,'FontWeight','bold');
ylabel('Validation Accuracy','FontSize',24,'FontWeight','bold');
 
box on
% grid on
% grid minor

yyaxis left 
b = bar(train_proportion, train_time, 'DisplayName', 'train time');
b.FaceColor = [0 0.447 0.7];
b.EdgeColor = 'none';

ylabel('Train Time (s)','FontSize',24,'FontWeight','bold');
lgd =legend('show');
lgd.Box = 'on';
% lgd.LineWidth = 1.5;
% lgd.FontSize = 18;
% lgd.FontWeight = 'normal';

hold off;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% roc_curve
% export the figure as .eps format will keep the figure in high resolution in latex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file_name = "/home/dadafly/program/CNN_LSTM/result/conv_3l_win_10_108_fc_rnn2_fc_1024_N_summary_075_train/conv_3l_win_10_108_fc_rnn2_fc_1024_N_summary_075_train.xlsx";
[eye_close, ~] 	= xlsread(file_name, 4);
[both_feet, ~] 	= xlsread(file_name, 5);
[both_fist, ~] 	= xlsread(file_name, 6);
[left_fist, ~] 	= xlsread(file_name, 7);
[right_fist, ~] 	= xlsread(file_name, 8);

eye_close_fpr = eye_close(:,1);
eye_close_tpr = eye_close(:,3);

both_feet_fpr = both_feet(:,1);
both_feet_tpr = both_feet(:,3);

both_fist_fpr = both_fist(:,1);
both_fist_tpr = both_fist(:,3);

left_fist_fpr = left_fist(:,1);
left_fist_tpr = left_fist(:,3);

right_fist_fpr = right_fist(:,1);
right_fist_tpr = right_fist(:,3);

figure;
hold on;

eye_close_plot 	= plot(eye_close_fpr, eye_close_tpr, '-', 'DisplayName', 'eye closed','LineWidth',3);
both_feet_plot 	= plot(both_feet_fpr, both_feet_tpr, '-', 'DisplayName', 'both feet','LineWidth',3);
both_fist_plot 	= plot(both_fist_fpr, both_fist_tpr, '-', 'DisplayName', 'both fists','LineWidth',3);
left_fist_plot 	= plot(left_fist_fpr, left_fist_tpr, '-', 'DisplayName', 'left fist','LineWidth',3);
right_fist_plot = plot(right_fist_fpr, right_fist_tpr, '-', 'DisplayName', 'right fist','LineWidth',3);
diagonal_plot = plot([0,0.5,1], [0,0.5,1], '--k', 'LineWidth',3);


set(gca, 'FontSize',24,'FontWeight','bold','linewidth',2); % set axes style
 
xlabel('False Positive Rate','FontSize',24,'FontWeight','bold');
ylabel('True Positive Rate','FontSize',24,'FontWeight','bold');

lgd =legend([eye_close_plot both_feet_plot both_fist_plot left_fist_plot right_fist_plot]);
lgd.Box = 'on';

box on


axes('position', [.35 .20 .45 .45])

hold on;
indexOfInterest_eye_close = (eye_close_fpr < 0.002 ) & (eye_close_fpr > 0);
indexOfInterest_both_feet = (both_feet_fpr < 0.002 ) & (both_feet_fpr > 0);
indexOfInterest_both_fist = (both_fist_fpr < 0.002 ) & (both_fist_fpr > 0);
indexOfInterest_left_fist = (left_fist_fpr < 0.002 ) & (left_fist_fpr > 0);
indexOfInterest_right_fist = (right_fist_fpr < 0.002 ) & (right_fist_fpr > 0);

plot(eye_close_fpr(indexOfInterest_eye_close),eye_close_tpr(indexOfInterest_eye_close), '-', 'DisplayName', 'eye closed','LineWidth',2) % plot on new axes
plot(both_feet_fpr(indexOfInterest_both_feet),both_feet_tpr(indexOfInterest_both_feet), '-', 'DisplayName', 'both feet','LineWidth',2) % plot on new axes
plot(both_fist_fpr(indexOfInterest_both_fist),both_fist_tpr(indexOfInterest_both_fist), '-', 'DisplayName', 'both fist','LineWidth',2) % plot on new axes
plot(left_fist_fpr(indexOfInterest_left_fist),left_fist_tpr(indexOfInterest_left_fist), '-', 'DisplayName', 'left fist','LineWidth',2) % plot on new axes
plot(right_fist_fpr(indexOfInterest_right_fist),right_fist_tpr(indexOfInterest_right_fist), '-', 'DisplayName', 'right fist','LineWidth',2) % plot on new axes
axis tight
set(gca,'FontSize',18,'FontWeight','bold','linewidth',2); % set axes style
box on
% lgd.LineWidth = 1.5;
% lgd.FontSize = 18;
% lgd.FontWeight = 'normal';

hold off;









