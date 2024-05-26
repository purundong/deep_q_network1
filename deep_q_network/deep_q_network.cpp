#include "deep_q_network.h"
#include <QScrollBar>
#include "state/ui_state.h"
#include <QMessageBox>
#include "neural_network.h"

deep_q_network::deep_q_network(QWidget* parent)
	: QWidget(parent),
	_plot{std::make_unique<QCustomPlot>()}
{
	ui.setupUi(this);
	QHBoxLayout* layout = new QHBoxLayout;
	layout->setContentsMargins(0, 0, 0, 0);
	ui.widget_loss->setLayout(layout);
	layout->addWidget(_plot.get());
	_plot->setContentsMargins(0, 0, 0, 0);
	connect(this, SIGNAL(sig_loss(double,double)), this, SLOT(sot_loss(double, double)), Qt::QueuedConnection);
	connect(this, SIGNAL(sig_msg_box(QString)), this, SLOT(sot_msg_box(QString)), Qt::QueuedConnection);
	connect(this, SIGNAL(sig_init(double, double, double, double)), this, SLOT(sot_init(double, double, double, double)), Qt::BlockingQueuedConnection);
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
	//menu_action = new QAction(QString::fromLocal8Bit("查看状态信息"));
	//connect(menu_action, SIGNAL(triggered()), this, SLOT(menu_state_info()));
	//_menu->addAction(menu_action);
	ui.horizontalSlider->blockSignals(true);
	ui.horizontalSlider->setPageStep(1);
	ui.horizontalSlider->setValue(_magni_fication);
	ui.horizontalSlider->setMinimum(1);
	ui.horizontalSlider->setMaximum(20);
	ui.horizontalSlider->blockSignals(false);
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
	auto map_state_obj = _environment->get_map_state();
	for (auto& s_key : *map_state_obj) {
		auto& fat = s_key.second->get_feature();
		auto* s_ui = new ui_state(s_key.second);
		ui.tableWidget->setCellWidget(fat._x, fat._y, s_ui);
		s_ui->show_state();
	}
}

void deep_q_network::on_pushButton_solve_clicked()
{
	if (_thr != nullptr) {
		_thr->interrupt();
		if (_thr->joinable())
			_thr->join();
		_thr = nullptr;
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
				neural_network_ptr main_network{ std::make_shared<neural_network>(neural_num) },
					target_network{ std::make_shared<neural_network>(neural_num) };
				main_network->to(*device);
				target_network->to(*device);
				boost::this_thread::interruption_point();
				torch::optim::SGDOptions opt(learning_rate);
				torch::optim::SGD sgd(main_network->parameters(), opt);
				auto mse_loss = torch::nn::MSELoss();
				auto trajectory_obj = _environment->sampling(step_count);
				unsigned int update_i = 1;
				for (int i = 0; i < replay_num; ++i) {
					for (int step_i = 0; step_i < step_count;) {
						auto replay_buf_obj = trajectory_obj->get_random_replay_buf(target_network, model_num, gama, *device);
						for (auto& buf_key : *replay_buf_obj) {
							boost::this_thread::interruption_point();
							sgd.zero_grad();
							auto loss = mse_loss(buf_key[0].target, main_network->forward(buf_key[0].data));
							loss.backward({ torch::ones_like(loss) }, true);
							sgd.step();
							if (update_i++ % 10)
								sig_loss(update_i, loss.item<double>());
						}
						step_i += model_num - 1;
						target_network->copy_params(*main_network);
					}
				}

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
				sig_msg_box(QString::fromStdString(e.msg()));
			}
			_thr = nullptr;
		};

	_thr = std::make_unique<boost::thread>(fun, device, neural_num, learning_rate, gama, error, step_count, model_num, replay_num);
}

void deep_q_network::sot_loss(double x, double y)
{
	auto* graph =_plot->graph(0);
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