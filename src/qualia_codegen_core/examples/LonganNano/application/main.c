#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include "nuclei_sdk_soc.h"
#include "drv_usb_hw.h"
#include "cdc_acm_core.h"

#include "NeuralNetwork.h"

extern uint8_t usb_data_buffer[CDC_ACM_DATA_PACKET_SIZE];
extern volatile uint8_t packet_sent, packet_receive;
extern volatile uint32_t receive_length;

enum state_t {
	INIT,
	RECEIVING,
	COMPUTING,
	RESULT
};

#define MAX_READ_SIZE 16384
uint8_t receive_buff[MAX_READ_SIZE] = {0}; //Define the receive array
uint16_t receive_buff_cnt = 0;

usb_core_driver USB_OTG_dev =
{
		.dev = {
				.desc = {
						.dev_desc       = (uint8_t *)&device_descriptor,
						.config_desc    = (uint8_t *)&configuration_descriptor,
						.strings        = usbd_strings,
				}
		}
};

int main(void)
{
	eclic_global_interrupt_enable();

	eclic_priority_group_set(ECLIC_PRIGROUP_LEVEL2_PRIO2);

	usb_rcu_config();

	usb_timer_init();

	usb_intr_config();

	usbd_init (&USB_OTG_dev, USB_CORE_ENUM_FS, &usbd_cdc_cb);

	/* check if USB device is enumerated successfully */
	while (USBD_CONFIGURED != USB_OTG_dev.dev.cur_status) {
	}
	delay_1ms(1000); // Wait for one second for host to initialize USB device and serial port
	receive_length = 0; // Make sure nothing is waiting in the buffer to be processed
	

	uint64_t start = SysTimer_GetLoadValue();
	int msglen = 0;
	char msg[64] = {0};
	enum state_t state = INIT;
	float *inputs;

	while (1) {
		if (USBD_CONFIGURED == USB_OTG_dev.dev.cur_status) {
			switch (state) {
			case INIT:
				if (receive_buff_cnt > 0) {
					state = RECEIVING;
					break;
				}
				if (SysTimer_GetLoadValue() - start > SOC_TIMER_FREQ/10) { // Periodically send READ until we receive something
					msglen = snprintf(msg, CDC_ACM_DATA_PACKET_SIZE, "READY\r\n");
					start = SysTimer_GetLoadValue();
				}
				break;
			case RECEIVING:
				if (receive_buff_cnt > 0 && receive_buff[receive_buff_cnt - 1] == '\n') {// Wait for EOL
					inputs = serialBufToFloats(receive_buff, receive_buff_cnt);
					msglen = snprintf(msg, 32, "%d\r\n", receive_buff_cnt);
					receive_buff_cnt = 0;

					state = COMPUTING;
					break;
				}
				break;
			case COMPUTING:
				if (packet_sent) { // wait for ack to be sent
					struct NNResult res = neuralNetworkInfer(inputs);
					msglen = snprintf(msg, 32, "%d,%d,%f\r\n", res.inference_count, res.label, (double)res.dist);
					state = RESULT;

				}
				break;
			case RESULT:
				if (packet_sent) { // wait for result to be sent
					state = RECEIVING;
				}
				break;
			}


			// === USB CDC -> UART ===
			if (packet_receive) {
				if (receive_length > 0 && receive_buff_cnt < MAX_READ_SIZE) {
					memcpy(receive_buff+receive_buff_cnt, usb_data_buffer, receive_length);
					receive_buff_cnt += receive_length;
				}
				cdc_acm_data_receive(&USB_OTG_dev);
			}

			// === UART -> USB CDC ===
			if (packet_sent) {
				if (msglen > 0) {
					memcpy(usb_data_buffer, msg, msglen);
					cdc_acm_data_send(&USB_OTG_dev, msglen);
					msglen = 0;
				}
			}
		}
	}

	return 0;
}

