#include "deep_q_network.h"
#include <QScrollBar>
#include "state/ui_state.h"
#include <QMessageBox>
#include "neural_network.h"

deep_q_network::deep_q_network(QWidget* parent)
	: QWidget(parent)
{
	ui.setupUi(this);
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
	auto neural_num = ui.spinBox_neural_num->value();
	auto learning_rate = ui.doubleSpinBox_learning_rate->value();
	auto gama = ui.doubleSpinBox_gama->value();
	auto error = ui.doubleSpinBox_error->value();
	auto step_count = ui.doubleSpinBox_error->value();
	auto model_num = ui.spinBox_model_num->value();
	auto replay_num = ui.spinBox_replay_num->value();


	std::unique_ptr<torch::Device> device;
	if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
		device = std::make_unique<torch::Device>(torch::kCUDA);
	}
	else {
		device = std::make_unique<torch::Device>(torch::kCPU);
		QMessageBox::warning(this, QString::fromLocal8Bit("警告"), 
			QString::fromLocal8Bit("你得设备不支持cuda以及cudnn加速,如果想要获得更好得性能请正确安装cuda12.4和对应的cudnn"), QMessageBox::Ok);
	}

	auto fun = [&](std::unique_ptr<torch::Device> device, unsigned int neural_num, float learning_rate, float gama,
		float error, size_t step_count, size_t model_num, int replay_num) {
		neural_network_ptr main_network{ std::make_shared<neural_network>(neural_num) },
			target_network{ std::make_shared<neural_network>(neural_num) };
		main_network->to(*device);
		target_network->to(*device);

		torch::optim::SGDOptions opt(learning_rate);
		torch::optim::SGD sgd(main_network->parameters(), opt);
		auto mse_loss = torch::nn::MSELoss();
		auto replay_buf_obj = _environment->sampling(step_count)->get_random_replay_buf(target_network, model_num, gama, *device);
			for (auto& buf_key : *replay_buf_obj) {
				for (int i = 0; i < buf_key.size(); ++i) {
					
					auto loss = mse_loss(buf_key[i].target, main_network->forward(buf_key[i].data));
					loss.backward();
				}

			}
			target_network->copy_params(*main_network);
		}

	};

}
