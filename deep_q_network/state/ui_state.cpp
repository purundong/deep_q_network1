#include "ui_state.h"
#include <QPainter>
#include <QGraphicsOpacityEffect>
#include <QColor>

ui_state::ui_state(state_ptr s_obj, QWidget* parent)
	: QWidget(parent), _state(s_obj)
{
	ui.setupUi(this);
	ui.widget_up->setAttribute(Qt::WA_WState_WindowOpacitySet);
	ui.widget_dow->setAttribute(Qt::WA_WState_WindowOpacitySet);
	ui.widget_right->setAttribute(Qt::WA_WState_WindowOpacitySet);
	ui.widget_left->setAttribute(Qt::WA_WState_WindowOpacitySet);
	ui.widget_fixed->setAttribute(Qt::WA_WState_WindowOpacitySet);
	show_state();
}

ui_state::ui_state(QWidget* parent)
	: QWidget(parent)
{
	ui.setupUi(this);
	show_state();
}

ui_state::~ui_state()
{
}

void ui_state::resizeEvent(QResizeEvent* event)
{
	show_state();
}

void ui_state::set_widget_color(const QColor& color)
{
	QPalette pal1(palette());
	setAutoFillBackground(true);
	pal1.setColor(QPalette::Base, color);
	setPalette(pal1);
}

void ui_state::show_state()
{
	{
		ui.widget_up->setAutoFillBackground(true);
		QImage image;
		QPalette palette;
		image.load(":/deep_q_network/arrow.png"); // 指定图片所在位置及图片名
		QTransform matrix;
		matrix.rotate(-90.0);
		palette.setBrush(this->backgroundRole(), QBrush(image.transformed(matrix, Qt::FastTransformation).scaled(ui.widget_up->size())));
		ui.widget_up->setPalette(palette);
		auto graph = new QGraphicsOpacityEffect(ui.widget_up);
		graph->setOpacity(_state->policy(action::feature::up));
		ui.widget_up->setGraphicsEffect(graph);
	}

	{
		ui.widget_right->setAutoFillBackground(true);
		QImage image;
		QPalette palette;
		image.load(":/deep_q_network/arrow.png"); // 指定图片所在位置及图片名
		palette.setBrush(this->backgroundRole(), QBrush(image.scaled(ui.widget_right->size())));
		ui.widget_right->setPalette(palette);
		auto graph = new QGraphicsOpacityEffect(ui.widget_up);
		graph->setOpacity(_state->policy(action::feature::right));
		ui.widget_right->setGraphicsEffect(graph);
	}

	{
		ui.widget_dow->setAutoFillBackground(true);
		QImage image;
		QPalette palette;
		image.load(":/deep_q_network/arrow.png"); // 指定图片所在位置及图片名
		QTransform matrix;
		matrix.rotate(90.0);
		palette.setBrush(this->backgroundRole(), QBrush(image.transformed(matrix, Qt::FastTransformation).scaled(ui.widget_dow->size())));
		ui.widget_dow->setPalette(palette);
		auto graph = new QGraphicsOpacityEffect(ui.widget_up);
		graph->setOpacity(_state->policy(action::feature::down));
		ui.widget_dow->setGraphicsEffect(graph);
	}

	{
		ui.widget_left->setAutoFillBackground(true);
		QImage image;
		QPalette palette;
		image.load(":/deep_q_network/arrow.png"); // 指定图片所在位置及图片名
		QTransform matrix;
		matrix.rotate(180);
		palette.setBrush(this->backgroundRole(), QBrush(image.transformed(matrix, Qt::FastTransformation).scaled(ui.widget_left->size())));
		ui.widget_left->setPalette(palette);
		auto graph = new QGraphicsOpacityEffect(ui.widget_up);
		graph->setOpacity(_state->policy(action::feature::left));
		ui.widget_left->setGraphicsEffect(graph);
	}

	{
		ui.widget_fixed->setAutoFillBackground(true);
		QImage image;
		QPalette palette;
		image.load(":/deep_q_network/circle.png"); // 指定图片所在位置及图片名
		palette.setBrush(this->backgroundRole(), QBrush(image.scaled(ui.widget_fixed->size())));
		ui.widget_fixed->setPalette(palette);
		auto graph = new QGraphicsOpacityEffect(ui.widget_up);
		graph->setOpacity(_state->policy(action::feature::fixed));
		ui.widget_fixed->setGraphicsEffect(graph);
	}

	auto rewrod = _state->reward_expectations();
	if (rewrod == 0)
		set_widget_color(QColor(255, 255, 255));
	else if (rewrod > 0)
		set_widget_color(QColor(51, 51, 255));
	else
		set_widget_color(QColor(153, 0, 76));
}
