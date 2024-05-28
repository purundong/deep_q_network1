#include "deep_q_network.h"
#include <QScrollBar>
#include "state/ui_state.h"
#include <QMessageBox>
#include "neural_network.h"

deep_q_network::deep_q_network(QWidget* parent)
	: QWidget(parent),
	_plot{ std::make_unique<QCustomPlot>() }
{
	ui.setupUi(this);
	QHBoxLayout* layout = new QHBoxLayout;
	layout->setContentsMargins(0, 0, 0, 0);
	ui.widget_loss->setLayout(layout);
	layout->addWidget(_plot.get());
	_plot->setContentsMargins(0, 0, 0, 0);

	connect(this, SIGNAL(sig_loss(double, double)), this, SLOT(sot_loss(double, double)), Qt::QueuedConnection);
	connect(this, SIGNAL(sig_msg_box(QString)), this, SLOT(sot_msg_box(QString)), Qt::QueuedConnection);
	connect(this, SIGNAL(sig_init(double, double, double, double)), this, SLOT(sot_init(double, double, double, double)), Qt::BlockingQueuedConnection);
	connect(this, SIGNAL(sig_show_environment()), this, SLOT(sot_show_environment()), Qt::QueuedConnection);
	init_tableWidget();
}

deep_q_network::~deep_q_network()
{
}

void deep_q_network::init_tableWidget()
{
	ui.tableWidget->verticalHeader()->hide();
	ui.tableWidget->horizontalHeader()->hide();
	ui.tableWidget->viewport()->installEventFilter(this);
	ui.tableWidget->setEditTriggers(QAbstractItemView::NoEditTriggers);
	ui.tableWidget->verticalHeader()->setDefaultSectionSize(_magni_fication * 10);
	ui.tableWidget->horizontalHeader()->setDefaultSectionSize(_magni_fication * 10);
	ui.tableWidget->setColumnWidth(0, 10);
	ui.tableWidget->setRowHeight(0, 10);
	ui.tableWidget->verticalScrollBar()->installEventFilter(this);
	ui.tableWidget->setContextMenuPolicy(Qt::CustomContextMenu);
	_menu = new QMenu(ui.tableWidget);

	QAction* menu_action = new QAction(QString::fromLocal8Bit("设置奖励"));
	connect(menu_action, SIGNAL(triggered()), this, SLOT(menu_set_rewards()));
	_menu->addAction(menu_action);
	menu_action = new QAction(QString::fromLocal8Bit("设置状态转移"));
	connect(menu_action, SIGNAL(triggered()), this, SLOT(menu_set_transfers()));
	_menu->addAction(menu_action);
	menu_action = new QAction(QString::fromLocal8Bit("查看状态信息"));
	connect(menu_action, SIGNAL(triggered()), this, SLOT(menu_state_info()));
	_menu->addAction(menu_action);
	ui.horizontalSlider->blockSignals(true);
	ui.horizontalSlider->setPageStep(1);
	ui.horizontalSlider->setValue(_magni_fication);
	ui.horizontalSlider->setMinimum(1);
	ui.horizontalSlider->setMaximum(20);
	ui.horizontalSlider->blockSignals(false);
}

void deep_q_network::show_environment()
{
	auto map_state_obj = _environment->get_map_state();
	for (auto& s_key : *map_state_obj) {
		auto& fat = s_key.second->get_feature();
		auto* s_ui = new ui_state(s_key.second);
		ui.tableWidget->setCellWidget(fat._x, fat._y, s_ui);
		s_ui->show_state();
	}
}

void deep_q_network::menu_set_rewards()
{
	int row = ui.tableWidget->currentRow();
	int col = ui.tableWidget->currentColumn();
	if (row < 0 || col < 0) return;
	auto* s_ui = dynamic_cast<ui_state*>(ui.tableWidget->cellWidget(row, col));
	if (s_ui == nullptr) return;
	s_ui->get_state()->make_set_reword()->exec();
	s_ui->show_state();
}

void deep_q_network::menu_set_transfers()
{
	int row = ui.tableWidget->currentRow();
	int col = ui.tableWidget->currentColumn();
	if (row < 0 || col < 0) return;
	auto* s_ui = dynamic_cast<ui_state*>(ui.tableWidget->cellWidget(row, col));
	if (s_ui == nullptr) return;
	s_ui->get_state()->make_set_state_transfers(_environment->get_map_state())->exec();
}

void deep_q_network::menu_state_info()
{
	int row = ui.tableWidget->currentRow();
	int col = ui.tableWidget->currentColumn();
	if (row < 0 || col < 0) return;
	auto* s_ui = dynamic_cast<ui_state*>(ui.tableWidget->cellWidget(row, col));
	if (s_ui == nullptr) return;
	s_ui->get_state()->make_show_state_info()->exec();
}

void deep_q_network::on_pushButton_make_state_clicked()
{
	int r_size = ui.spinBox_row->value();
	int c_size = ui.spinBox_col->value();
	int trap = ui.spinBox_trap->value();
	int target = ui.spinBox_target->value();
	int all = r_size * c_size;

	if (all < (trap + target)) {
		QMessageBox::warning(this, QString::fromLocal8Bit("错误"), QString::fromLocal8Bit("陷阱和终点之和不得大于状态数"), QMessageBox::Ok);
		return;
	}

	ui.tableWidget->setRowCount(0);
	ui.tableWidget->setColumnCount(0);

	ui.tableWidget->setRowCount(r_size);
	ui.tableWidget->setColumnCount(c_size);
	_environment = std::make_shared<environment>(r_size, c_size, trap, target);
	show_environment();
}

void deep_q_network::on_pushButton_dqn_clicked()
{
	if (_thr != nullptr) {
		_thr->interrupt();
		if (_thr->joinable())
			_thr->join();
		_thr = nullptr;
		return;
	}

	if (_environment == nullptr || _environment->empty()) {
		sig_msg_box(QString::fromLocal8Bit("没有状态"));
		return;
	}


	auto neural_num = ui.spinBox_neural_num->value();
	auto learning_rate = ui.doubleSpinBox_learning_rate->value();
	auto gama = ui.doubleSpinBox_gama->value();
	auto error = ui.doubleSpinBox_error->value();
	auto step_count = ui.spinBox_step_count->value();
	auto model_num = ui.spinBox_model_num->value();
	auto replay_num = ui.spinBox_replay_num->value();


	std::shared_ptr<torch::Device> device;
	if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
		device = std::make_unique<torch::Device>(torch::kCUDA);
	}
	else {
		device = std::make_unique<torch::Device>(torch::kCPU);
		QMessageBox::warning(this, QString::fromLocal8Bit("警告"),
			QString::fromLocal8Bit("你得设备不支持cuda以及cudnn加速,如果想要获得更好得性能请正确安装cuda12.4和对应的cudnn"), QMessageBox::Ok);
	}

	auto fun = [&](std::shared_ptr<torch::Device> device, unsigned int neural_num, float learning_rate, float gama,
		float error, size_t step_count, size_t model_num, int replay_num) {
			try
			{
				sig_init(0.0, step_count * replay_num, 0.0, 1);
				neural_network_ptr main_network{ std::make_shared<neural_network>(neural_num) },//定义main_network, neural_num为隐藏神经元成个数
					target_network{ std::make_shared<neural_network>(neural_num) };//定义target_network, neural_num为隐藏神经元成个数
				main_network->to(*device);
				target_network->to(*device);
				boost::this_thread::interruption_point();
				torch::optim::AdamW sgd(main_network->parameters(), learning_rate);//建立优化器设置学习率 learning_rate为学习率
				auto mse_loss = torch::nn::MSELoss();//定义损失函数(Qpi(s,a)-Q(s,a))^2

				auto trajectory_obj = _environment->sampling(step_count); //在环境中走step_count步

				unsigned int update_i = 0;
				for (int i = 0; i < replay_num; ++i) {//遍历回放次数
					auto replay_buf_obj = trajectory_obj->get_random_replay_buf(*device); //生成一个打乱顺序的replay_buf;
					for (auto& buf_key : *replay_buf_obj) { //遍历这个打乱顺序的replay_buf

						boost::this_thread::interruption_point();


						bool is = torch::equal(buf_key[0].target.slice(0, 1, 3).view({ 1,2 }), buf_key[0].data.view({ 1,3 }).slice(1, 0, 2));

						auto q = torch::max(target_network->forward(buf_key[0].target.slice(0, 1, 16).view({ 5,3 }))); 
						auto td_target = (buf_key[0].target[0].view({ 1, 1 }) + (gama * q)).view({ 1,1 });
						
						auto y = main_network->forward(buf_key[0].data.view({ 1,3 })); //预测当前Q(st,a)

						auto loss = mse_loss(td_target, y); //使用MES计算LOSS 其实就是计算TDERROR 这里使用的是你的j(w)函数利用自动求导的方式计算梯度


						sgd.zero_grad(); //清空神经网络梯度

						loss.backward({ torch::ones_like(loss) }, true); //计算梯度

						sgd.step();//更新参数main_network的权重
						if ((update_i % 10) == 0) //每10次迭代输出一次loss
							sig_loss(update_i, loss.item<double>());

						if ((update_i % model_num) == 0)//每model_num次迭代更新一次target_network
							target_network->copy_params(*main_network);
						update_i++;
					}
				}
				_environment->update_agent(main_network, *device);
				sig_show_environment();
				sig_msg_box(QString::fromLocal8Bit("完成"));
			}
			catch (const std::runtime_error& e) {
				auto a = e.what();
				sig_msg_box(QString::fromStdString(e.what()));
			}
			catch (const boost::thread_interrupted& e)
			{
			}
			catch (const at::Error& e)
			{
				auto a = e.what();
				sig_msg_box(QString::fromStdString(e.msg()));
			}
			_thr = nullptr;
		};

	_thr = std::make_unique<boost::thread>(fun, device, neural_num, learning_rate, gama, error, step_count, model_num, replay_num);
}

void deep_q_network::on_pushButton_qlearning_clicked()
{
	if (_thr != nullptr) {
		_thr->interrupt();
		if (_thr->joinable())
			_thr->join();
		_thr = nullptr;
		return;
	}

	if (_environment == nullptr || _environment->empty()) {
		sig_msg_box(QString::fromLocal8Bit("没有状态"));
		return;
	}

	auto neural_num = ui.spinBox_neural_num->value();
	auto learning_rate = ui.doubleSpinBox_learning_rate->value();
	auto gama = ui.doubleSpinBox_gama->value();
	auto error = ui.doubleSpinBox_error->value();
	auto step_count = ui.spinBox_step_count->value();
	auto model_num = ui.spinBox_model_num->value();
	auto replay_num = ui.spinBox_replay_num->value();

	auto fun = [&](std::shared_ptr<torch::Device> device, unsigned int neural_num, float learning_rate, float gama,
		float error, size_t step_count, size_t model_num, int replay_num) {
			auto trajectory_obj = _environment->sampling(step_count);
			for (int i = 0; i < replay_num; ++i) {
				auto& samples = trajectory_obj->get_samples();
				for (auto begin = samples.begin(); begin != samples.end(); ++begin) {
					double td_target = begin->_reword + gama * begin->_next_state->max_max_action_value();
					auto td_error = begin->_curr_action->value() - td_target;
					auto a_value = begin->_curr_action->value() - 0.005 * (td_error);
					begin->_curr_action->set_value(a_value);
				}
			}
			_environment->update_state_policy_value();
			sig_show_environment();
			sig_msg_box(QString::fromLocal8Bit("完成"));
		};
	_thr = std::make_unique<boost::thread>(fun, nullptr, neural_num, learning_rate, gama, error, step_count, model_num, replay_num);
}

void deep_q_network::sot_loss(double x, double y)
{
	auto* graph = _plot->graph(0);
	graph->addData(x, y);
	_plot->replot();
	_plot->show();
}

void deep_q_network::sot_init(double x_min, double x_max, double y_min, double y_max)
{
	_plot->clearPlottables(); // 移除所有可绘制对象（曲线等）
	_plot->clearItems(); // 移除所有辅助项（比如文本标签）
	_plot->xAxis->setLabel(QObject::tr("次数"));
	_plot->xAxis->setLabel(QObject::tr("loss"));
	_plot->xAxis->setRange(x_min, x_max);
	_plot->yAxis->setRange(y_min, y_max);
	_plot->addGraph();
	_plot->replot();
	_plot->show();
}

void deep_q_network::sot_msg_box(QString msg)
{
	QMessageBox::warning(this, QString::fromLocal8Bit("错误"), msg, QMessageBox::Ok);
}

void deep_q_network::sot_show_environment()
{
	show_environment();
}

void deep_q_network::on_horizontalSlider_valueChanged(int value)
{
	if (_magni_fication == value) return;
	_magni_fication = value;
	ui.tableWidget->verticalHeader()->setDefaultSectionSize(_magni_fication * 10);
	ui.tableWidget->horizontalHeader()->setDefaultSectionSize(_magni_fication * 10);
}

void deep_q_network::on_tableWidget_customContextMenuRequested(QPoint point)
{
	_menu->exec(QCursor::pos());
}