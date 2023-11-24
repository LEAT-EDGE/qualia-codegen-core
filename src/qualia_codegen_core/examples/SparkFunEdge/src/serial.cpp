// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

#include "serial.h"
extern "C" {
#include "am_bsp.h"  // NOLINT
#include "am_util_stdio.h"
}

#include <cstdlib>

#define UART_IDENT 0
#define RECEIVE_BUFFER_SIZE            32768 // Must be able to hold at least one full message of max length
#define SEND_BUFFER_SIZE            AM_PRINTF_BUFSIZE // We don't send back that much data, AM_PRINTF_BUFSIZE=256
#define READ_BLOCK_SIZE 32
static void *g_pvUART;
static uint8_t g_pui8UARTTXBuffer[SEND_BUFFER_SIZE];
static char g_psWriteData[RECEIVE_BUFFER_SIZE];
static char g_prfbuf[AM_PRINTF_BUFSIZE];
volatile uint32_t g_ui32UARTRxIndex = 0;
//volatile bool g_bRxTimeoutFlag = false;

void enable_burst_mode(void) {
  am_hal_burst_avail_e          eBurstModeAvailable;
  am_hal_burst_mode_e           eBurstMode;

  //
  // Check that the Burst Feature is available.
  //
  if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_initialize(&eBurstModeAvailable)) {
#if 0 //DEBUG
    if (AM_HAL_BURST_AVAIL == eBurstModeAvailable) {
      myapp_printf("Apollo3 Burst Mode is Available\n");
    } else {
      myapp_printf("Apollo3 Burst Mode is Not Available\n");
    }
  } else {
    myapp_printf("Failed to Initialize for Burst Mode operation\n");
#endif
  }

  //
  // Put the MCU into "Burst" mode.
  //
  if (AM_HAL_STATUS_SUCCESS == am_hal_burst_mode_enable(&eBurstMode)) {
#if 0 //DEBUG
    if (AM_HAL_BURST_MODE == eBurstMode) {
      myapp_printf("Apollo3 operating in Burst Mode (96MHz)\n");
    }
  } else {
    myapp_printf("Failed to Enable Burst Mode operation\n");
#endif
  }
}

void uart_init(void) {
//  enable_burst_mode();

  // Start the UART.
  am_hal_uart_config_t sUartConfig = {
    //
    // Standard UART settings: 115200-8-N-1
    //
    //.ui32BaudRate    = 115200,
    .ui32BaudRate    = 921600,
    .ui32DataBits    = AM_HAL_UART_DATA_BITS_8,
    .ui32Parity      = AM_HAL_UART_PARITY_NONE,
    .ui32StopBits    = AM_HAL_UART_ONE_STOP_BIT,
    .ui32FlowControl = AM_HAL_UART_FLOW_CTRL_NONE,

    // Set TX and RX FIFOs to interrupt at three-quarters full.
    .ui32FifoLevels = (AM_HAL_UART_TX_FIFO_1_4 |
                       AM_HAL_UART_RX_FIFO_1_4),

    // This code will use the standard interrupt handling for UART TX, but
    // we will have a custom routine for UART RX.
    .pui8TxBuffer = g_pui8UARTTXBuffer,
    .ui32TxBufferSize = sizeof(g_pui8UARTTXBuffer),
    .pui8RxBuffer = 0,
    .ui32RxBufferSize = 0,
  };

  am_hal_uart_initialize(UART_IDENT, &g_pvUART);
  am_hal_uart_power_control(g_pvUART, AM_HAL_SYSCTRL_WAKE, false);
  am_hal_uart_configure(g_pvUART, &sUartConfig);
  am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_TX, g_AM_BSP_GPIO_COM_UART_TX);
  am_hal_gpio_pinconfig(AM_BSP_GPIO_COM_UART_RX, g_AM_BSP_GPIO_COM_UART_RX);

  // Make sure to enable the interrupts for RX, since the HAL doesn't already
  // know we intend to use them.
  NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + UART_IDENT));
  am_hal_uart_interrupt_enable(g_pvUART, (AM_HAL_UART_INT_RX |
                               AM_HAL_UART_INT_RX_TMOUT));

  am_hal_interrupt_master_enable();


  //am_util_stdio_printf_init(myapp_uart_string_print);
}


//*****************************************************************************
//
//! @brief UART-based string print function.
//!
//! This function is used for printing a string via the UART, which for some
//! MCU devices may be multi-module.
//!
//! @return None.
//
//*****************************************************************************
void uart_string_print(char *pcString) {
    uint32_t ui32StrLen = 0;
    uint32_t ui32BytesWritten = 0;

    // Measure the length of the string.
    while (pcString[ui32StrLen] != 0) {
        ui32StrLen++;
    }

    // Print the string via the UART.
    const am_hal_uart_transfer_t sUartWrite = {
        .ui32Direction = AM_HAL_UART_WRITE,
        .pui8Data = (uint8_t *) pcString,
        .ui32NumBytes = ui32StrLen,
        .ui32TimeoutMs = AM_HAL_UART_WAIT_FOREVER,
        .pui32BytesTransferred = &ui32BytesWritten,
    };

    am_hal_uart_transfer(g_pvUART, &sUartWrite);

} // am_bsp_uart_string_print()

uint32_t printf(const char *pcFmt, ...) {
  uint32_t ui32NumChars;

  // Convert to the desired string.
  va_list pArgs;
  va_start(pArgs, pcFmt);
  ui32NumChars = am_util_stdio_vsprintf(g_prfbuf, pcFmt, pArgs);
  va_end(pArgs);

  // This is where we print the buffer to the configured interface. 
  uart_string_print(g_prfbuf);

  // return the number of characters printed.
  return ui32NumChars; 
}

//*****************************************************************************
//
// Interrupt handler for the UART.
//
//*****************************************************************************
extern "C" {
__attribute__((__used__))
void am_uart_isr(void)
{
  uint32_t ui32Status = 0;

  //
  // Read the masked interrupt status from the UART.
  //
  am_hal_uart_interrupt_status_get(g_pvUART, &ui32Status, true);
  am_hal_uart_interrupt_clear(g_pvUART, ui32Status);
  am_hal_uart_interrupt_service(g_pvUART, ui32Status, 0);


  //
  // If there's an RX interrupt, handle it in a way that preserves the
  // timeout interrupt on gaps between packets.
  //
  if (ui32Status & (AM_HAL_UART_INT_RX_TMOUT | AM_HAL_UART_INT_RX))
  {
      uint32_t ui32BytesRead = 0;
    if (g_ui32UARTRxIndex + READ_BLOCK_SIZE <= RECEIVE_BUFFER_SIZE) { // buffer has free space

      am_hal_uart_transfer_t sRead =
      {
        .ui32Direction = AM_HAL_UART_READ,
        .pui8Data = (uint8_t *) &(g_psWriteData[g_ui32UARTRxIndex]),
        //.pui8Data = (uint8_t *) &(g_psWriteData[0]),
        .ui32NumBytes = READ_BLOCK_SIZE,
        .ui32TimeoutMs = 0,
        .pui32BytesTransferred = &ui32BytesRead,
        //.pui32BytesTransferred = NULL,
      };

      am_hal_uart_transfer(g_pvUART, &sRead);
      //ui32BytesRead=0;

      g_ui32UARTRxIndex += ui32BytesRead;
    }
    //myapp_printf("%d %d %d %d\r\n", g_ui32UARTRxIndex, READ_BLOCK_SIZE, RECEIVE_BUFFER_SIZE, ui32BytesRead);

    //
    // If there is a TMOUT interrupt, assume we have a compete packet, and
    // send it over SPI.
    //
    //if (ui32Status & (AM_HAL_UART_INT_RX_TMOUT))
    //{
      //NVIC_DisableIRQ((IRQn_Type)(UART0_IRQn + UART_IDENT));
      //cmd_handler(g_psWriteData, g_ui32UARTRxIndex);
      //g_bRxTimeoutFlag = true;
    //}
  }

}
}

int serialBufToFloats(float input[]) {
  /*if (!g_bRxTimeoutFlag) {
    return -1;
  }*/

  /*if (g_bRxTimeoutFlag) {
    g_bRxTimeoutFlag = false;
    NVIC_EnableIRQ((IRQn_Type)(UART0_IRQn + UART_IDENT));
  }*/

  //myapp_printf("lastchar %c %d\r\n", g_psWriteData[g_ui32UARTRxIndex - 1], g_psWriteData[g_ui32UARTRxIndex - 1]);

  if (g_ui32UARTRxIndex < 1 || g_psWriteData[g_ui32UARTRxIndex - 1] != '\n') {
    // cannot rely only on timeout if sender is too slow, so check EOL, also make sure we got some data
    return 0;
  }


/*
  for (unsigned int i = 0; i < size; i++) {
    buf[i] = g_psWriteData[i];
  }
*/
  auto *pbuf = g_psWriteData;

  unsigned int i = 0;
  while ((pbuf - g_psWriteData) < (int)g_ui32UARTRxIndex && *pbuf != '\r' && *pbuf != '\n') {
    input[i] = strtof(pbuf, &pbuf);
    //printf("Parsed %d: %f, nextc: %c\r\n", i, input[i], *pbuf);
    i++;
    pbuf++;//skip delimiter
  }

  auto t = g_ui32UARTRxIndex;
  g_ui32UARTRxIndex = 0;

  return t;
}

